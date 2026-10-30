"""CPU-only check of the actual probe's literal-check allocation behavior."""
import argparse
import json
from pathlib import Path
import re
import tempfile

from compare_four_models import ROOT, sha, write
from test_model_selection import toolchain, run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    probe=ROOT/'tests/native/four_model_probe.cpp'
    declarations=re.findall(r'^static void require\(bool good, const [^\n]+',probe.read_text(),re.M)
    if len(declarations)!=2 or not any('const char*' in line for line in declarations):
        raise ValueError('Unexpected probe check overloads')
    reference=next(line for line in declarations if 'std::string&' in line).replace('require(','reference(',1)
    source=r'''
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
static unsigned long long allocations=0;
void* operator new(size_t bytes) { ++allocations; if(auto p=malloc(bytes))return p;throw std::bad_alloc(); }
void operator delete(void* p) noexcept { free(p); }
void operator delete(void* p,size_t) noexcept { free(p); }
'''+reference+'\n'+'\n'.join(declarations)+r'''
int main() {
    constexpr unsigned components=200000;
    volatile unsigned code=0x3c00;
    using Clock=std::chrono::steady_clock;
    auto started=Clock::now(); allocations=0;
    for(unsigned i=0;i<components;++i)reference((code&0x7c00)!=0x7c00,"Nonfinite model output");
    auto old_count=allocations;
    double old_ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    started=Clock::now(); allocations=0;
    for(unsigned i=0;i<components;++i)require((code&0x7c00)!=0x7c00,"Nonfinite model output");
    auto new_count=allocations;
    double new_ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
    bool literal_error=false,dynamic_error=false;
    try { require(false,"Nonfinite model output"); }
    catch(const std::runtime_error& error) { literal_error=strcmp(error.what(),"Nonfinite model output")==0; }
    std::string dynamic_message="A dynamic path failure message";
    try { require(false,dynamic_message); }
    catch(const std::runtime_error& error) { dynamic_error=error.what()==dynamic_message; }
    bool passed=old_count==components && new_count==0 && literal_error && dynamic_error;
    printf("{\"passed\":%s,\"components\":%u,\"old_allocations\":%llu,\"new_allocations\":%llu,\"old_cpu_ms\":%.6f,\"new_cpu_ms\":%.6f,\"failure_messages_preserved\":%s}\n",
           passed?"true":"false",components,old_count,new_count,old_ms,new_ms,literal_error&&dynamic_error?"true":"false");
    return passed?0:1;
}
'''
    compiler,env=toolchain(64)
    with tempfile.TemporaryDirectory(prefix='probe-checks-',dir=ROOT/'build') as temporary:
        work=Path(temporary)
        path=work/'checks.cpp';path.write_text(source)
        executable=work/'checks.exe'
        run([compiler,'/nologo','/std:c++20','/O2','/EHsc','/MT',path,
             '/Fo:'+str(work/'checks.obj'),'/Fe:'+str(executable)],cwd=work,env=env)
        report=json.loads(run([executable],cwd=work,env=env))
    report.update(gpu_used=False,probe_source_sha256=sha(probe),temporary_builds_removed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write(args.output,report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
