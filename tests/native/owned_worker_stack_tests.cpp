// Reproduce legacy executables whose default stack is smaller than model setup.
#define NOMINMAX
#include "../../src/integration/compatibility/owned_worker.h"
#include <future>
#include <iostream>
#include <stdexcept>
#pragma comment(linker,"/STACK:196608")

__declspec(noinline) unsigned touch_worker_stack() {
    volatile unsigned char pages[512*1024];
    unsigned sum=0;
    for(unsigned i=0;i<sizeof(pages);i+=4096){pages[i]=static_cast<unsigned char>(i/4096);sum+=pages[i];}
    return sum;
}
int main()try{
    struct State{std::promise<unsigned> result;};
    auto state=std::make_shared<State>();auto result=state->result.get_future();
    std::weak_ptr<State> weak=state;
    phoenix_compat::start_owned_worker(state,[](State& value){value.result.set_value(touch_worker_stack());});
    state.reset();
    if(result.wait_for(std::chrono::seconds(5))!=std::future_status::ready || result.get()!=8128)
        throw std::runtime_error("Bounded worker stack workload failed");
    const auto deadline=GetTickCount64()+1000;
    while(!weak.expired()&&GetTickCount64()<deadline)Sleep(1);
    if(!weak.expired())throw std::runtime_error("Worker lease did not retire");
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"small_host_stack\":196608,\"worker_stack\":"
             <<phoenix_compat::processing_worker_stack_bytes<<"}";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
