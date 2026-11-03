#include "../../src/integration/compatibility/capture_queue.h"
#include <iostream>
#include <stdexcept>
using phoenix_compat::CaptureQueue;
int main()try {
    unsigned checks=0;auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("Capture queue regression");};
    auto rejects=[&](auto action){bool rejected=false;try{action();}catch(const std::logic_error&){rejected=true;}check(rejected);};
    CaptureQueue queue;check(queue.capacity()==1&&queue.free_slot()==0&&!queue.front());
    rejects([&]{queue.pop();});rejects([&]{queue.capacity(3);});
    queue.capacity(2);queue.push({10,100,1000,1,false});queue.push({11,101,1010,0,false});
    check(queue.free_slot()==-1&&queue.size()==2);
    rejects([&]{queue.capacity(1);});rejects([&]{queue.pop();});
    check(queue.front()->frame==100&&queue.front()->slot==1);
    queue.front()->retired=true;
    // CPU readback alone does not release the paired GPU input texture.
    check(queue.free_slot()==-1);queue.pop();
    check(queue.free_slot()==1&&queue.front()->frame==101&&queue.front()->tick==1010);
    rejects([&]{queue.push({12,102,1020,0,false});});
    rejects([&]{queue.push({10,102,1020,1,false});});
    queue.push({12,102,1020,1,false});queue.front()->retired=true;queue.pop();
    check(queue.front()->frame==102&&queue.front()->serial==12);
    queue.front()->retired=true;queue.pop();check(!queue.front()&&queue.free_slot()==0);
    // A capture fence two presentations behind must not halve throughput when
    // inference can finish within one presentation interval. Test both budgets.
    auto simulate=[&](unsigned capacity) {
        CaptureQueue pending;pending.capacity(capacity);unsigned submitted=0;
        for(uint64_t frame=1;frame<=120;++frame) {
            if(auto* capture=pending.front();capture&&capture->frame+2<=frame) {
                check(capture->tick==capture->frame*17);capture->retired=true;pending.pop();++submitted;
            }
            auto slot=pending.free_slot();if(slot>=0)pending.push({frame,frame,frame*17,unsigned(slot),false});
        }
        return submitted;
    };
    check(simulate(1)==59);check(simulate(2)==118);
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"checks\":"<<checks<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
