#include "../../src/integration/compatibility/capture_queue.h"
#include "../../src/integration/compatibility/d3d9_inference_pacing.h"
#include <iostream>
#include <stdexcept>
using phoenix_compat::CaptureQueue;
int main()try {
    unsigned checks=0;auto check=[&](bool value){++checks;if(!value)throw std::runtime_error("Capture queue regression");};
    auto rejects=[&](auto action){bool rejected=false;try{action();}catch(const std::logic_error&){rejected=true;}check(rejected);};
    CaptureQueue queue;check(queue.capacity()==1&&queue.free_slot()==0&&!queue.front());
    rejects([&]{queue.pop();});rejects([&]{queue.capacity(CaptureQueue::max_slots+1);});
    rejects([&]{queue.capacity(0);});
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
    // Deeper queues drain in order and free every slot again.
    check(simulate(4)==118);
    {
        CaptureQueue in_flight;
        in_flight.capacity(2);
        in_flight.push({1,10,100,0,true});
        in_flight.pop(true);
        in_flight.push({2,11,110,1,true});
        in_flight.pop(true);
        check(in_flight.free_slot()==-1);
        rejects([&]{in_flight.push({3,12,120,0,false});});
        in_flight.release_snapshot(0);
        check(in_flight.free_slot()==0);
        rejects([&]{in_flight.release_snapshot(0);});
        in_flight.release_snapshot(1);
    }
    {
        CaptureQueue deep;deep.capacity(4);
        for(unsigned i=0;i<4;++i)deep.push({20+i,200+i,300+i,i,false});
        check(deep.size()==4&&deep.free_slot()==-1);
        for(unsigned i=0;i<4;++i){check(deep.front()->frame==200+i);deep.front()->retired=true;deep.pop();}
        check(!deep.front()&&deep.free_slot()==0);
    }
    {
        CaptureQueue latest;
        latest.capacity(4);
        latest.push({1,10,100,0,true});
        latest.pop(true);
        for (unsigned i = 1; i <= 3; ++i) latest.push({1+i,10+i,100+i,i,false});
        check(latest.discard_superseded(2)==0);
        check(latest.discard_superseded(3)==1 && latest.front()->frame==12 && latest.free_slot()==1);
        latest.front()->retired=true;
        check(latest.discard_superseded(4)==0 && latest.front()->frame==12);
        latest.pop(true);
        check(latest.discard_superseded(4)==0 && latest.front()->slot==3);
        latest.release_snapshot(0);
        latest.release_snapshot(2);
        check(latest.free_slot()==0);
    }
    {
        phoenix_compat::D3D9InferencePacing pacing;
        auto window = [&](double period, unsigned passes, bool started) {
            for (unsigned i = 0; i < 24; ++i) pacing.observe(period, passes, started);
        };
        window(13.33, 2, false);
        check(pacing.gap_us()==2000);
        check(pacing.frame_budget_us()>=13320 && pacing.frame_budget_us()<=13340);
        window(13.34, 2, true);
        check(pacing.gap_us()==1750);
        window(15.0, 2, true);
        check(pacing.gap_us()==2750);
        window(14.0, 2, true);
        check(pacing.gap_us()==3250);
        for (unsigned i=0;i<8;++i) window(16.0,2,true);
        check(pacing.gap_us()==6500);
        window(13.33, 1, true);
        check(pacing.gap_us()==2000);
        pacing.observe(400.0,1,true);
        check(pacing.gap_us()==2000 && pacing.frame_budget_us()==0);
    }
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"checks\":"<<checks<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
