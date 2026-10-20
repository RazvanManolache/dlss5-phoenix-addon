"""Matched released/current native inference with the authenticated older DLL.

Both source snapshots and kernel builds are disposable. Only compact evidence
survives; external DLLs, installed games and deployed addons remain untouched.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from benchmark_readme_case import fixture
from compare_four_models import ROOT, build, sha, telemetry, write

OLD_DLL_SHA256 = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
NEW_DLL_SHA256 = '2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f'


def common_probe(source):
    """Adapt only the probe, leaving each snapshot's production code intact.

    The old release lacks model selection and adaptive scheduling. Fixed work
    avoids timing two different benchmark frontends while retaining each actual
    production recorder, dispatch implementation, graph and compiled kernels.
    """
    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise ValueError('Probe adapter anchor changed: '+old[:65])
        source = source.replace(old, new)
    replace('#include "../../native_runtime/network_schedule.h"\n', '')
    replace('phoenix_cpp::NetworkSchedule schedule;schedule.prepare(config,source.model_id,true);',
        'struct { unsigned passes=1, depth=8; std::array<unsigned,8> repeats{{1,1,1,1,1,1,1,1}}; } schedule;\n'
        '        require(config.at("weight_source").value("model",std::string("original"))=="original", "Original model only");\n'
        '        require(config.value("passes",1u)==1 && config.value("transformer_blocks",8u)==8 &&\n'
        '            config.value("repeats",std::array<unsigned,8>{{1,1,1,1,1,1,1,1}})==schedule.repeats &&\n'
        '            !config.contains("adaptive_budget"), "Full fixed trained schedule required");')
    replace('schedule.prepare(config,source.model_id,true);', '')
    replace('schedule.complete({{UINT32_MAX,0,pair.second,true,UINT32_MAX}});', '')
    replace('schedule.status()', 'Json{{"transformer_blocks",8},{"passes",1}}')
    replace('bundle->model_resource.name', 'std::string("WEIGHTS_HT")')
    source = source.replace('source.model_id', 'std::string("original")')
    # Deep GPU markers were introduced after 0.82; this comparison deliberately
    # uses only the same two total-time timestamps on both builds.
    start = '        if(deep_gpu){\n            diagnostic_heap='
    end = '        auto frame='
    if source.count(start) != 1:
        raise ValueError('Deep-marker adapter anchor changed')
    a = source.index(start)
    b = source.index(end, a)
    source = source[:a]+source[b:]
    replace('const bool profile_gpu=config.value("profile_gpu_stages",false);',
        'require(!config.value("profile_gpu_stages",false), "Uninstrumented comparison only");\n'
        '        const bool profile_gpu=false;')
    return source


def snapshot(ref, destination):
    """Read committed sources without modifying branches or making a worktree."""
    raw = subprocess.check_output(['git','archive','--format=zip',ref], cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            target = (destination/name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Source archive path escaped temporary directory')
        archive.extractall(destination)
    return hashlib.sha256(raw).hexdigest()


def round_order(cells, repeat):
    return list(cells) if repeat % 2 == 0 else list(reversed(cells))


def summarize(cases):
    groups = {}
    for case in cases:
        groups.setdefault((case['build'],case['dll'],case['engine'],*case['extent']),[]).append(case)
    rows = []
    for (label,dll,engine,w,h), entries in groups.items():
        row = dict(build=label,dll=dll,engine=engine,extent=[w,h],processes=len(entries))
        for field in ('gpu','wall'):
            values = [entry[field]['median_ms'] for entry in entries]
            row[field] = dict(mean_process_median_ms=statistics.mean(values),
                process_medians_ms=values,process_range_ms=[min(values),max(values)])
        row['create_ms'] = [entry['create_ms'] for entry in entries]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released-ref', default='origin/main')
    parser.add_argument('--old-dll', type=Path, default=ROOT.parent/'_dlls/nvngx_dlssnr.dll')
    parser.add_argument('--new-dll', type=Path,
        default=ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/current/release-ab-3108')
    args = parser.parse_args()
    old_ref = subprocess.check_output(['git','rev-parse',args.released_ref+'^{commit}'],cwd=ROOT,text=True).strip()
    current_ref = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    models = {'310.8': args.old_dll.resolve(), '310.9.16': args.new_dll.resolve()}
    for name, expected in [('310.8',OLD_DLL_SHA256),('310.9.16',NEW_DLL_SHA256)]:
        if sha(models[name]) != expected:
            raise ValueError('Wrong authenticated model DLL: '+name)
    cores = list(Path('C:/Windows/System32/DriverStore/FileRepository').glob('nv_*.inf_amd64_*/_nvngx.dll'))
    if len(cores) != 1:
        raise ValueError('Need one unambiguous installed NGX core')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    report_path = output/'run.json'
    if report_path.exists():
        raise ValueError('Use a fresh result folder; do not mix build identities')
    template = subprocess.check_output(['git','show',current_ref+':tests/native/four_model_probe.cpp'],cwd=ROOT,text=True)
    harness = common_probe(template)
    report = dict(passed=False,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        games_launched=False,installed_files_modified=False,model_files_modified=False,
        source_refs={'released':old_ref,'current':current_ref},
        common_harness_sha256=hashlib.sha256(harness.encode()).hexdigest(),
        model_dll_sha256={'310.8':OLD_DLL_SHA256,'310.9.16':NEW_DLL_SHA256},
        model='original',passes=1,transformer_blocks=8,repeats=[1]*8,
        fixture='Current structured SDR RGB fixture quantized through FP16; zero motion; no depth; fixed seed; automatic mask; default tone/structure',
        gpu_before=telemetry(),builds={},cases=[],warmup_seconds=1.,measurement_seconds=3.,
        scope='Common GPU-resident D3D12 probe; engine work and matching output copy; upload/readback excluded; full call records/submits/waits',
        limitations=['Clocks and OS scheduling are not locked.',
            'Released means current origin/main source, not the earlier commit that produced historical README numbers.',
            'Fixed Original-only benchmark frontend; not addon UI, transport, game FPS or startup-to-first-image.'])
    write(report_path,report)
    references = {}
    try:
        with tempfile.TemporaryDirectory(prefix='release_ab_',dir=ROOT/'build') as temporary:
            work = Path(temporary)
            builds = {}
            for label,ref in report['source_refs'].items():
                source = work/label
                source.mkdir()
                archive_hash = snapshot(ref,source)
                # A generated identical harness is the only substituted source.
                probe = source/'tests/native/four_model_probe.cpp'
                probe.parent.mkdir(parents=True,exist_ok=True)
                probe.write_text(harness,encoding='utf-8',newline='\n')
                print('BUILD '+label+' '+ref[:10],flush=True)
                binary = source/'build/probe'
                binary.mkdir(parents=True)
                executable,forwarder = build(binary,source_root=source)
                subprocess.run([sys.executable,source/'tools/production_build_native_kernels.py',
                    'matched','--compiler','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe',
                    '--jobs','4'],cwd=source,check=True,timeout=1800)
                bundle = source/'build/matched'
                manifest = json.loads((bundle/'native_manifest.json').read_text())
                builds[label] = dict(executable=executable,forwarder=forwarder,bundle=bundle,work=binary)
                report['builds'][label] = dict(source_archive_sha256=archive_hash,probe_sha256=sha(executable),
                    forwarder_sha256=sha(forwarder),native_manifest_sha256=sha(bundle/'native_manifest.json'),
                    native_graph_sha256=manifest['native_graph_sha256'],kernels=[
                        {key:row[key] for key in ('name','entry','binary_sha256')} for row in manifest['kernels']])
                write(report_path,report)
            for w,h,rounds in ((1280,720,2),(1920,1080,4),(3840,2160,2)):
                pixels = np.ones((h,w,4),np.float16)
                pixels[...,:3] = fixture(w,h)
                color,motion = work/'color.rgba16f',work/'motion.rg16f'
                color.write_bytes(pixels.tobytes())
                motion.write_bytes(np.zeros((h,w,2),np.float16).tobytes())
                cells = [(label,'310.8',engine) for engine in ('phoenix','nvidia') for label in ('released','current')]
                if w == 1920:
                    cells += [('current','310.9.16',engine) for engine in ('phoenix','nvidia')]
                for repeat in range(rounds):
                    for label,dll,engine in round_order(cells,repeat):
                        built = builds[label]
                        name = f'{label}-{dll}-{engine}-{w}x{h}-p{repeat}'
                        config = dict(width=w,height=h,engine=engine,weight_source=dict(kind='nvidia_dll',
                            path=str(models[dll]),sha256=report['model_dll_sha256'][dll],model='original'),
                            bundle_path=str(built['bundle']),ngx_core=str(cores[0]),forwarder=str(built['forwarder']),
                            cache=str(built['work']/'cache'),color_file=str(color),motion_file=str(motion),
                            debug=False,capture=False,warmup_seconds=1.,measure_seconds=3.)
                        job = work/'job.json'
                        destination = output/(name+'.json')
                        write(job,config)
                        before = telemetry()
                        process = subprocess.run([built['executable'],job,destination],cwd=built['work'],
                            capture_output=True,text=True,timeout=180)
                        result = json.loads(destination.read_text()) if destination.exists() else {'passed':False}
                        result.update(process_exit_code=process.returncode)
                        if process.returncode or not result.get('passed'):
                            report['failed_case'] = name
                            report['failure'] = result.get('error',process.stderr[-2000:])
                            raise RuntimeError('Isolated comparison failed: '+name)
                        captures = {name:value['sha256'] for name,value in result['captures'].items()}
                        reference_key = (dll,w,h)
                        reference = references.setdefault(reference_key,captures)
                        if captures != reference:
                            report['failed_case'] = name
                            report['failure'] = 'Reset/history capture mismatch against matched reference'
                            raise RuntimeError('Output mismatch: '+name)
                        record = dict(build=label,dll=dll,engine=engine,extent=[w,h],repeat=repeat,
                            gpu=result['gpu'],wall=result['wall'],create_ms=result['create_ms'],
                            memory=result['memory'],captures=captures,input_sha256=result['input_sha256'],
                            observed_launches=result.get('neural_launches',result.get('observed_cuda_launches')),
                            result_file=destination.name,result_sha256=sha(destination),
                            telemetry_before=before,telemetry_after=telemetry())
                        report['cases'].append(record)
                        write(report_path,report)
                        print(name+' PASS GPU '+str(round(result['gpu']['median_ms'],6))+' ms',flush=True)
            report['summary'] = summarize(report['cases'])
            report['passed'] = True
        report['temporary_builds_removed'] = True
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['gpu_after'] = telemetry()
        write(report_path,report)
    print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']),summary=report['summary']),indent=2))


if __name__ == '__main__':
    main()
