"""Explicit isolated native-kernel tuning experiments at 1080p.

Candidate sources and kernels live in one temporary workspace. Production
sources/packages are never changed by this screen. Full graphs, default
controls and before/after reset/history hashes are checked for both models.
"""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import numpy as np
from compare_four_models import ROOT, build, sha, telemetry, write
from production_build_native_kernels import configure_boundary_ptx
from test_model_selection import toolchain, run

CUDA = Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin')
PRODUCTION = ROOT/'src/native/kernels/production'
BASE = ROOT/'artifacts/current/native-kernel-styles'
MODEL = ROOT.parent/'research/model-compatibility-310.9.16/nvngx_dlssnr.310.9.16.dll'
STYLES = {'auto':None, 'rtx40':890, 'rtx30':860, 'rtx20':750}
HOT = ('input/preparation.cu', 'output/reconstruction.cu',
       'window_attention/channels_32/attention.cu', 'window_attention/channels_128/attention.cu',
       'window_attention/channels_256/attention.cu', 'global_attention/transformer/attention.cu',
       'global_attention/transformer/feed_forward_contract.cu')


def replace_once(path, before, after):
    text = path.read_text()
    if text.count(before) != 1:
        raise ValueError(f'Expected exactly one source anchor in {path.name}')
    path.write_text(text.replace(before, after))


def source_variant(folder, profile):
    """Create a candidate without touching the maintained production tree."""
    import shutil
    shutil.copytree(PRODUCTION, folder)
    math = folder/'common/architecture_math.cuh'
    if profile == 'packed_decode':
        replace_once(math,
            '    return uint32_t(decode_fp8_value(uint8_t(bytes))) |\n'
            '           (uint32_t(decode_fp8_value(uint8_t(bytes >> 8))) << 16);',
            '''#if defined(__CUDA_ARCH__)
    const uint32_t low = bytes & 127u, high = (bytes >> 8) & 127u;
    const uint32_t signs = ((bytes & 0x80u) << 8) | ((bytes & 0x8000u) << 16);
    const uint32_t magnitudes = low | (high << 16);
    const uint32_t normals = (magnitudes << 7) + 0x20002000u;
    const uint32_t selector = ((low & 7u) << 4) | ((high & 7u) << 12);
    const uint32_t subnormals = __byte_perm(0x1e1c1800u, 0x23222120u, selector);
    const uint32_t lo = low == 127 ? 0x7e00u : ((low < 8 ? subnormals : normals) & 0xffffu);
    const uint32_t hi = high == 127 ? 0x7e000000u : ((high < 8 ? subnormals : normals) & 0xffff0000u);
    return signs | lo | hi;
#else
    return uint32_t(decode_fp8_value(uint8_t(bytes))) |
           (uint32_t(decode_fp8_value(uint8_t(bytes >> 8))) << 16);
#endif''')
    elif profile in ('weight_value_cache','global_weight_pairs','global_weight_values'):
        # E4M3 has only 256 possible decoded values. Cache that vocabulary once
        # per GPU module instead of expanding/copying proprietary model data.
        def half(code):
            m, sign = code & 127, (code & 128) << 8
            if m == 127: return sign | 0x7e00
            if m >= 8: return sign | ((m << 7) + 0x2000)
            return sign | [0,0x1800,0x1c00,0x1e00,0x2000,0x2100,0x2200,0x2300][m]
        table = ','.join(hex(half(i)) for i in range(256))
        text = math.read_text()
        anchor = '// Fragment order follows mma.m16n8k16.'
        text = text.replace(anchor,
            '__device__ __constant__ uint16_t cached_weight_values[256] = {'+table+'};\n'
            '__device__ __forceinline__ void cached_weight_fragment(uint32_t word, uint32_t& even, uint32_t& odd) {\n'
            ' even=uint32_t(cached_weight_values[word&255]) | (uint32_t(cached_weight_values[(word>>16)&255])<<16);\n'
            ' odd=uint32_t(cached_weight_values[(word>>8)&255]) | (uint32_t(cached_weight_values[word>>24])<<16);\n'
            '}\n'+anchor)
        calls = ('decode_fp8_fragment(b[i], right_low[i], right_high[i]);',
                 'decode_fp8_weight_fragment(b[i], right_low[i], right_high[i]);')
        matches = [call for call in calls if call in text]
        if len(matches) != 1 or text.count(matches[0]) != 1:
            raise ValueError('Weight decoder anchor changed; refuse a mislabeled cache experiment')
        text = text.replace(matches[0], 'cached_weight_fragment(b[i], right_low[i], right_high[i]);')
        if profile=='global_weight_values':
            text=text.replace('__device__ __constant__ uint16_t cached_weight_values',
                              '__device__ const uint16_t cached_weight_values')
        if profile=='global_weight_pairs':
            begin=text.index('__device__ __constant__ uint16_t cached_weight_values')
            end=text.index(anchor,begin)
            # A global, read-only 16-bit vocabulary avoids the divergent
            # constant-memory broadcast path. This is model-independent data.
            pairs=','.join(hex(half(i&255) | (half(i>>8)<<16))+'u' for i in range(65536))
            text=text[:begin]+('__device__ const uint32_t cached_weight_pairs[65536] = {'+pairs+'};\n'
                '__device__ __forceinline__ void cached_weight_fragment(uint32_t word,uint32_t& even,uint32_t& odd) {\n'
                ' even=cached_weight_pairs[(word&255)|((word>>8)&0xff00)];\n'
                ' odd=cached_weight_pairs[((word>>8)&255)|((word>>16)&0xff00)];\n'
                '}\n')+text[end:]
        math.write_text(text)
    elif profile == 'late_fragments':
        text = math.read_text()
        weight_decoder = ('decode_fp8_weight_fragment' if
                          'decode_fp8_weight_fragment(b[i]' in text else 'decode_fp8_fragment')
        start = text.index('    const uint32_t a[4] = {a0, a1, a2, a3}, b[2] = {b0, b1};')
        end = text.index('\n}\n#endif', start)
        replacement = '''    uint32_t x0,y0,x1,y1,r0,r1;
    decode_fp8_fragment(a0,x0,y0); decode_fp8_fragment(a1,x1,y1);
    decode_fp8_fragment(b0,r0,r1);
    half_matrix(d0,d1,x0,x1,y0,y1,r0,r1,c0,c1);
    decode_fp8_fragment(a2,x0,y0); decode_fp8_fragment(a3,x1,y1);
    decode_fp8_fragment(b1,r0,r1);
    half_matrix(d0,d1,x0,x1,y0,y1,r0,r1,d0,d1);'''
        for operand in ('b0', 'b1'):
            replacement = replacement.replace('decode_fp8_fragment('+operand,
                                              weight_decoder+'('+operand)
        text = text[:start]+replacement+text[end:]
        math.write_text(text)
    elif profile == 'deferred_copy_wait':
        memory = folder/'common/compatibility_copy.cuh'
        if 'void finish_compatibility_copies()' in memory.read_text():
            raise ValueError('Deferred waits are already implemented. Use the earlier pinned source snapshot for that historical experiment.')
        replace_once(memory,
            '    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");',
            '    asm volatile("cp.async.commit_group;" ::: "memory");')
        text = memory.read_text()
        marker = 'namespace phoenix_architecture {\n'
        text = text.replace(marker, marker+'''// Complete each issuing thread's copies at the existing CTA consumer barrier.
__device__ __forceinline__ void finish_compatibility_copies() {
#if PHOENIX_STYLE_ARCH >= 800 && PHOENIX_STYLE_ARCH < 900
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
}
''', 1)
        memory.write_text(text)
        for path in folder.rglob('*.cu'):
            text = path.read_text()
            if 'bulk_copy<' in text:
                path.write_text(text.replace('__syncthreads();',
                    'phoenix_architecture::finish_compatibility_copies();'))
    elif profile == 'vector_memory':
        path = folder/'common/device_memory_scalar.cuh'
        for cache in ('ca','cg'):
            before = f'''    const uint32_t* word_pointer = reinterpret_cast<const uint32_t*>(address);
    address_value = __ld{cache}(word_pointer);
    word_y = __ld{cache}(word_pointer + 1);
    word_z = __ld{cache}(word_pointer + 2);
    weights = __ld{cache}(word_pointer + 3);'''
            after = f'''    const uint4 packed = __ld{cache}(reinterpret_cast<const uint4*>(address));
    address_value=packed.x; word_y=packed.y; word_z=packed.z; weights=packed.w;'''
            replace_once(path,before,after)
    elif profile == 'ordered_mma':
        path = folder/'common/feed_forward_math.cuh'
        text = path.read_text().replace('    asm("mma.sync.', '    asm volatile("mma.sync.')
        path.write_text(text)
    else:
        raise ValueError('Unknown experiment '+profile)


def selected_contracts(contracts, profile):
    if profile=='all_math':
        return [c for c in contracts if '/layout/' not in c['source']]
    if profile in ('global_weight_pairs','global_weight_values'):
        return [c for c in contracts if any(c['source'].endswith('/'+name) for name in HOT[:2]+HOT[4:5])]
    if profile == 'deferred_copy_wait':
        return [c for c in contracts if 'bulk_copy<' in (ROOT/c['source']).read_text()]
    if profile == 'vector_memory':
        return [c for c in contracts if 'device_memory_scalar.cuh' in (ROOT/c['source']).read_text()]
    return [c for c in contracts if any(c['source'].endswith('/'+name) for name in HOT)]


def compile_case(source, bundle, profile, style, contracts, env):
    """Replace only the screened modules; other production binaries stay pinned."""
    manifest = json.loads((BASE/'native_manifest.json').read_text())
    selected = manifest['kernels'] if style == 'auto' else manifest['kernel_styles'][style]['kernels']
    bundle.mkdir()
    # Hard links share immutable control bytes; overrides always have new paths.
    paths = {row['binary'] for row in manifest['kernels']}
    paths |= {r['binary'] for v in manifest['kernel_styles'].values() for r in v['kernels']}
    for name in ('native_graph.json','native_graph_v2.json'):
        if (BASE/name).exists(): paths.add(name)
    for name in paths:
        target = bundle/name; target.parent.mkdir(parents=True,exist_ok=True)
        os.link(BASE/name,target)
    targets = selected_contracts(contracts,profile)
    def compile_one(contract):
        relative = Path(contract['source']).relative_to('src/native/kernels/production')
        target = bundle/'overrides'/relative.with_suffix(''); target.mkdir(parents=True)
        binary,ptx = target/'kernel.cubin',target/'kernel.ptx'
        options = contract.get('ptx_options')
        entries=[contract['entry']]+([contract['large_input_entry']] if 'large_input_entry' in contract else [])
        command=[CUDA/'nvcc.exe','--entries',','.join(entries),'-allow-unsupported-compiler',
                 '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH','-diag-suppress','177','-diag-suppress','550',
                 '-ptx' if options else '-cubin','-arch=sm_120','-O3','-std=c++17','-lineinfo',
                 *([f'-DPHOENIX_KERNEL_STYLE={STYLES[style]}'] if style!='auto' else []),
                 *([] if options else [f'-Xptxas=-O{contract.get("assembler_optimization",3)},--suppress-debug-info']),
                 source/relative,'-o',ptx if options else binary]
        run(command,cwd=bundle,env=env,timeout=600)
        if options:
            configure_boundary_ptx(ptx,'sm_120',contract['entry'],options)
            run([CUDA/'ptxas.exe','-arch=sm_120','--entry',contract['entry'],
                 f'-O{contract.get("assembler_optimization",3)}','--suppress-debug-info',ptx,'-o',binary],
                cwd=bundle,env=env,timeout=600)
        return contract['name'],binary
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,binary in pool.map(compile_one,targets):
            row=next(r for r in selected if r['name']==name)
            row.update(binary=binary.relative_to(bundle).as_posix(),binary_sha256=sha(binary),
                       architecture_images=[dict(architecture='sm_120',sha256=sha(binary),bytes=binary.stat().st_size)])
    if style!='auto':manifest['kernel_styles'][style]['architectures']=['sm_120']
    write(bundle/'native_manifest.json',manifest)
    return [c['name'] for c in targets]


def fixture(work):
    height,width=1080,1920
    y,x=np.mgrid[:height,:width].astype(np.float32)
    pixels=np.ones((height,width,4),np.float16)
    pixels[...,0]=x/width; pixels[...,1]=y/height
    pixels[...,2]=.5+.25*np.sin(x*.17)*np.cos(y*.11)
    (work/'color.rgba16f').write_bytes(pixels.tobytes())
    (work/'motion.rg16f').write_bytes(np.zeros((height,width,2),np.float16).tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',nargs='+',default=['packed_decode','weight_value_cache','late_fragments',
                         'deferred_copy_wait','vector_memory','ordered_mma'])
    parser.add_argument('--rounds',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/kernel-tuning')
    parser.add_argument('--wait-for',type=Path)
    parser.add_argument('--all-modules',action='store_true',help='Apply the candidate to every arithmetic module, not only the hot-module screen')
    args=parser.parse_args()
    destination=args.output.resolve();destination.mkdir(parents=True,exist_ok=True)
    report_path=destination/'screen.json'
    report=dict(passed=False,state='running',games_launched=False,installed_files_modified=False,
                older_device_validated=False,extent=[1920,1080],gpu_before=telemetry(),
                baseline_manifest_sha256=sha(BASE/'native_manifest.json'),cases=[],decisions=[])
    write(report_path,report)
    contracts=json.loads((ROOT/'src/native/model/production_kernel_contracts.json').read_text())['kernels']
    _,env=toolchain(64)
    styles={'packed_decode':['rtx20','rtx30'],'weight_value_cache':['rtx20','rtx30'],
            'late_fragments':['rtx20','rtx30'],'deferred_copy_wait':['rtx30','rtx40'],
            'global_weight_pairs':['rtx20','rtx30'],
            'global_weight_values':['rtx20','rtx30'],
            'vector_memory':['auto','rtx40','rtx30','rtx20'],'ordered_mma':['auto','rtx40']}
    try:
        with tempfile.TemporaryDirectory(prefix='native-tuning-',dir=ROOT/'build') as temporary:
            work=Path(temporary);executable,forwarder=build(work);fixture(work)
            report['probe_sha256']=sha(executable)
            if args.wait_for:
                while json.loads(args.wait_for.read_text()).get('state')=='running':time.sleep(1)
            for profile in args.profiles:
                source=work/('source-'+profile);source_variant(source,profile)
                for style in styles[profile]:
                    key=profile+'-'+style;folder=work/key
                    print('Compiling',key,flush=True)
                    changed=compile_case(source,folder,'all_math' if args.all_modules else profile,style,contracts,env)
                    report.setdefault('candidate_manifests',{})[key]=json.loads((folder/'native_manifest.json').read_text())
                    for model in ('original','small-fp8'):
                        measurements={'control':[],'candidate':[]}
                        for repeat in range(args.rounds):
                            for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                                config=dict(width=1920,height=1080,engine='phoenix',
                                    weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=sha(MODEL),model=model),
                                    bundle_path=str(BASE if label=='control' else folder),native_kernel_style=style,
                                    forwarder=str(forwarder),color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                                    capture=False,debug=False,warmup_seconds=.5,measure_seconds=1.,minimum_measured_frames=30)
                                write(work/'job.json',config);(work/'result.json').unlink(missing_ok=True)
                                process=subprocess.run([str(executable),str(work/'job.json'),str(work/'result.json')],
                                    cwd=work,capture_output=True,text=True,timeout=90)
                                result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {}
                                if process.returncode or not result.get('passed'):
                                    row=dict(profile=profile,style=style,model=model,label=label,round=repeat,passed=False,
                                             error=result.get('error',process.stderr))
                                else:
                                    row=dict(profile=profile,style=style,model=model,label=label,round=repeat,passed=True,
                                             gpu=result['gpu'],wall=result['wall'],captures=result['captures'],
                                             reset_repeatable=result['reset_repeatable'],memory=result['memory'],
                                             gpu_samples_ms=result['gpu_samples_ms'])
                                    measurements[label].append(row)
                                report['cases'].append(row);write(report_path,report)
                                if not row['passed']:
                                    print(key,model,label,'FAILED',row['error'],flush=True)
                                    break
                            else:continue
                            break
                        good=all(len(rows)==args.rounds for rows in measurements.values())
                        exact=good and all(row['captures']==measurements['control'][0]['captures']
                                           for rows in measurements.values() for row in rows)
                        decision=dict(profile=profile,style=style,model=model,complete=good,exact=exact,
                                      changed_kernels=changed,screen_scope='Selected hot modules in the complete model graph')
                        if good:
                            before,after=[statistics.median(r['gpu']['median_ms'] for r in measurements[k]) for k in ('control','candidate')]
                            decision.update(control_ms=before,candidate_ms=after,change_percent=(after/before-1)*100,
                                            control_rounds_ms=[r['gpu']['median_ms'] for r in measurements['control']],
                                            candidate_rounds_ms=[r['gpu']['median_ms'] for r in measurements['candidate']])
                        report['decisions'].append(decision);write(report_path,report)
                        print(key,model,'exact',exact,'change',decision.get('change_percent'),flush=True)
            exact = all(row.get('exact',False) for row in report['decisions'])
            report.update(passed=exact,evaluation_completed=True,all_candidates_exact=exact,
                          state='completed',gpu_after=telemetry())
    except Exception as error:
        report.update(state='failed',error=str(error));raise
    finally:
        report['temporary_builds_removed']=True;write(report_path,report)


if __name__=='__main__':main()
