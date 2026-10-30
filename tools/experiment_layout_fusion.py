"""Isolated producer/repack fusion screen. No maintained runtime is patched.

The producer keeps its original output and additionally writes token layout.
A benchmark-only forwarding shim omits the redundant repack GPU dispatch while
retaining the existing schedule audit and group barrier. Both models run full
graphs; exact reset/history equality is required before considering adoption.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from compare_four_models import ROOT, build, sha, write, telemetry
from experiment_native_tuning import BASE, MODEL, STYLES, CUDA, fixture
from test_model_selection import toolchain, run

KERNEL='cc_split_swin_16h_final_head_512_wait_fp8'
SOURCE='optimized/global_attention/split_heads_512/final_head_wait.cu'


def prepare_shadow(shadow):
    for name in ('src','native_runtime','third_party'):
        shutil.copytree(ROOT/name,shadow/name)
    (shadow/'tests/native').mkdir(parents=True)
    for name in ('four_model_probe.cpp','kernel_timing.h'):
        shutil.copy2(ROOT/'tests/native'/name,shadow/'tests/native'/name)
    # Expose only in this throwaway executable; production API stays untouched.
    dispatch=shadow/'src/native/support/dispatch.h'
    dispatch.write_text(dispatch.read_text().replace('class FrameDispatch {','class FrameDispatch {\npublic:',1))
    policy=shadow/'src/native/support/network_policy.h'
    policy_text=policy.read_text().replace('namespace policy_detail {',
        'namespace policy_detail {\ninline bool experimental_fused_layout=false;',1)
    policy_text=policy_text.replace('add(split + "final_head_512_wait_fp8", 40, 8);',
        'add(split + "final_head_512_wait_fp8", experimental_fused_layout?48:40, 8);')
    # Extend only this candidate's preflight contract; the normal 40-byte ABI
    # remains active for the paired control in the same executable.
    anchor='    return bindings;'
    assert policy_text.count(anchor)==1
    policy_text=policy_text.replace(anchor,
        '    if(experimental_fused_layout && kernel_name=="'+KERNEL+'")\n'
        '        bindings.pointers.push_back({40,"output_tokens"});\n'+anchor)
    policy.write_text(policy_text)
    probe=shadow/'tests/native/four_model_probe.cpp'
    text=probe.read_text()
    text=text.replace('destination=argv[2];',
        'destination=argv[2];phoenix_native::policy_detail::experimental_fused_layout=config.value("benchmark_fuse_layout",false);',1)
    anchor='static LaunchFn real_launch=nullptr;'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
static LaunchFn fusion_forward=nullptr;
static decltype(NVAPI_CU_KERNEL_LAUNCH_PARAMS{}.hFunction) fusion_skip{};
static unsigned long long fusion_omitted=0;
static NvAPI_Status __cdecl fusion_dispatch(ID3D12GraphicsCommandList* commands,
    const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches,NvU32 count) {
    if(count==1 && launches[0].hFunction==fusion_skip) { ++fusion_omitted; return NVAPI_OK; }
    return fusion_forward(commands,launches,count);
}
namespace phoenix_native {
struct NetworkDiagnostics {
    static auto repack_handle(const Network& network,unsigned position) {
        return network.programs_[0].at(position).launch.hFunction;
    }
};
}
''')
    anchor='report["neural_launches"]=network->program_steps();report["model_resource"]=bundle->model_resource.name;'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
            if(config.value("benchmark_fuse_layout",false)) {
                const auto layout=phoenix_model::native_layout(plan.value("model_architecture",std::string("phoenix-network37/1")));
                fusion_skip=phoenix_native::NetworkDiagnostics::repack_handle(*network,layout.repack_in());
                fusion_forward=dispatch->real_; dispatch->real_=fusion_dispatch;
                report["actual_neural_launches_per_pass"]=network->program_steps()-1;
            }
''')
    anchor='report["debug_errors"]=gpu.errors();'
    text=text.replace(anchor,'report["fused_repack_dispatches_omitted"]=fusion_omitted;'+anchor)
    probe.write_text(text)
    source=shadow/'src/native/kernels/production'/SOURCE
    text=source.read_text().replace('uint32_t words[10];','uint32_t words[12];')
    text=text.replace('sizeof(KernelArguments) == 40','sizeof(KernelArguments) == 48')
    text=text.replace('store_four_words_streaming<CheckBounds>(', 'fused_store<CheckBounds>(parameters, ')
    anchor='template <bool CheckBounds>\n__device__ __forceinline__ void run_global_attention_split_heads_512_final_head_wait('
    assert text.count(anchor)==1
    helper='''#include "../../../common/feature_layout.cuh"
template<bool CheckBounds> __device__ __forceinline__ void fused_store(
    const uint32_t* parameters,uint64_t address,uint32_t a,uint32_t b,uint32_t c,uint32_t d,
    const MemoryRange* ranges,uint32_t* errors) {
    store_four_words_streaming<CheckBounds>(address,a,b,c,d,ranges,errors);
    const uint64_t base=combine_words(parameters[2],parameters[3]);
    auto* tokens=reinterpret_cast<uint32_t*>(combine_words(parameters[10],parameters[11]));
    const uint32_t width=parameters[9],height=parameters[8];
    const uint32_t values[4]={a,b,c,d};
#pragma unroll
    for(unsigned i=0;i<4;++i) {
        const uint32_t word=uint32_t((address-base)/4)+i;
        const uint32_t tile=word/4096,local=word%4096;
        const uint32_t lane=((local>>4)&7) | ((local&1)<<3);
        const uint32_t feature=(local/128)*8+((local/4)&3)+(((local/2)&1)*4);
        const uint32_t row=(tile/(width/4))*4+lane/4;
        const uint32_t column=(tile%(width/4))*4+lane%4;
        if(row<height)tokens[phoenix::token_word_offset(row*width+column,feature)]=values[i];
    }
}
'''
    text=text.replace(anchor,helper+anchor)
    anchor='    run_global_attention_split_heads_512_final_head_wait<false>(parameters.words, nullptr, nullptr);'
    assert text.count(anchor)==1
    text=text.replace(anchor,'''    // Padding has no producer cell. Initialize it before the group barrier.
    if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0) {
        const uint32_t cells=parameters.words[8]*parameters.words[9];
        const uint32_t padded=(cells+31)/32*32;
        auto* target=reinterpret_cast<uint32_t*>(combine_words(parameters.words[10],parameters.words[11]));
        const unsigned thread=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        for(unsigned i=thread;i<(padded-cells)*256;i+=blockDim.x*blockDim.y*blockDim.z)
            target[phoenix::token_word_offset(cells+i/256,i%256)]=0;
    }
'''+anchor)
    source.write_text(text)


def candidate(shadow,work,style,env):
    folder=work/('bundle-'+style);folder.mkdir()
    manifest=json.loads((BASE/'native_manifest.json').read_text())
    paths={r['binary'] for r in manifest['kernels']}
    paths|={r['binary'] for v in manifest['kernel_styles'].values() for r in v['kernels']}
    for name in paths:
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(BASE/name,target)
    graph=json.loads((BASE/'native_graph.json').read_text())
    layer=next(l for l in graph['layers'] if l['kernel']==KERNEL)
    layer['bindings']['40']=dict(role='output_tokens',target='tensor:057.output')
    graph['tensors']['tensor:057.output']['producer']=layer['index']
    write(folder/'native_graph.json',graph);manifest['native_graph_sha256']=sha(folder/'native_graph.json')
    for row in manifest['kernels']+[r for v in manifest['kernel_styles'].values() for r in v['kernels']]:
        if row['name']==KERNEL:row['parameter_bytes']=48
    row=next(r for r in (manifest['kernels'] if style=='auto' else manifest['kernel_styles'][style]['kernels']) if r['name']==KERNEL)
    binary=folder/'fused.cubin'
    run([CUDA/'nvcc.exe','--entries',row['entry'],'-allow-unsupported-compiler',
         '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH','-cubin','-arch=sm_120','-O3','-std=c++17',
         *([f'-DPHOENIX_KERNEL_STYLE={STYLES[style]}'] if style!='auto' else []),
         '-Xptxas=-O3,--suppress-debug-info',shadow/'src/native/kernels/production'/SOURCE,'-o',binary],
        cwd=work,env=env,timeout=600)
    row.update(binary='fused.cubin',binary_sha256=sha(binary),architecture_images=[dict(architecture='sm_120',sha256=sha(binary),bytes=binary.stat().st_size)])
    if style!='auto':manifest['kernel_styles'][style]['architectures']=['sm_120']
    write(folder/'native_manifest.json',manifest)
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/current/kernel-tuning/fusion.json')
    parser.add_argument('--rounds',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--wait-for-screen',type=Path,
                        help='Compile first, then wait for this tuning screen to finish before GPU work')
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,state='running',scope='Fused producer/token-layout writes and removal of one GPU dispatch',
                games_launched=False,installed_files_modified=False,older_device_validated=False,extent=[1920,1080],
                gpu_before=telemetry(),cases=[],decisions=[])
    write(args.output,report);_,env=toolchain(64)
    try:
        with tempfile.TemporaryDirectory(prefix='layout-fusion-',dir=ROOT/'build') as tmp:
            work=Path(tmp);shadow=work/'shadow';prepare_shadow(shadow)
            executable,forwarder=build(work,source_root=shadow);fixture(work)
            prepared={}
            for style in ('auto','rtx40','rtx30','rtx20'):
                prepared[style]=candidate(shadow,work,style,env);print('Built fusion',style,flush=True)
            if args.wait_for_screen:
                while json.loads(args.wait_for_screen.read_text()).get('state')=='running':
                    time.sleep(1)
            report['gpu_before']=telemetry()
            for style in ('auto','rtx40','rtx30','rtx20'):
                folder=prepared[style]
                for model in ('original','small-fp8'):
                    groups={'control':[],'candidate':[]}
                    for repeat in range(args.rounds):
                        for label in (('control','candidate') if repeat%2==0 else ('candidate','control')):
                            config=dict(width=1920,height=1080,engine='phoenix',
                                weight_source=dict(kind='nvidia_dll',path=str(MODEL),sha256=sha(MODEL),model=model),
                                bundle_path=str(BASE if label=='control' else folder),native_kernel_style=style,
                                forwarder=str(forwarder),color_file=str(work/'color.rgba16f'),motion_file=str(work/'motion.rg16f'),
                                capture=False,debug=False,warmup_seconds=.5,measure_seconds=1.,minimum_measured_frames=30,
                                benchmark_fuse_layout=label=='candidate')
                            write(work/'job.json',config);(work/'result.json').unlink(missing_ok=True)
                            p=subprocess.run([str(executable),str(work/'job.json'),str(work/'result.json')],cwd=work,
                                             capture_output=True,text=True,timeout=90)
                            result=json.loads((work/'result.json').read_text()) if (work/'result.json').exists() else {}
                            row=dict(style=style,model=model,label=label,round=repeat,passed=not p.returncode and result.get('passed',False))
                            if row['passed']:
                                row.update(gpu=result['gpu'],captures=result['captures'],memory=result['memory'],
                                           omitted_repack_dispatches=result['fused_repack_dispatches_omitted'])
                                groups[label].append(row)
                            else:row['error']=result.get('error',p.stderr)
                            report['cases'].append(row);write(args.output,report)
                            if not row['passed']:print('Fusion failure',style,model,row.get('error'),flush=True);break
                        else:continue
                        break
                    good=all(len(g)==args.rounds for g in groups.values())
                    exact=good and all(r['captures']==groups['control'][0]['captures'] for g in groups.values() for r in g)
                    decision=dict(style=style,model=model,complete=good,exact=exact)
                    if good:
                        a,b=[statistics.median(r['gpu']['median_ms'] for r in groups[k]) for k in ('control','candidate')]
                        decision.update(control_ms=a,candidate_ms=b,change_percent=(b/a-1)*100)
                    report['decisions'].append(decision);write(args.output,report)
                    print('Fusion',style,model,decision,flush=True)
            exact = all(row.get('exact',False) for row in report['decisions'])
            report.update(passed=exact,evaluation_completed=True,all_candidates_exact=exact,state='completed')
    except Exception as error:report.update(state='failed',error=str(error));raise
    finally:report['temporary_builds_removed']=True;write(args.output,report)


if __name__=='__main__':main()
