#include "../../src/integration/compatibility/runtime_continuity.h"
#include <iostream>
#include <stdexcept>

struct Session {
    unsigned* retired;
    unsigned helper_pid = 17, completed_frames = 42;
    bool swapchain_views_released = false;
    explicit Session(unsigned& count) : retired(&count) {}
    ~Session() { ++*retired; }
};
int main() try {
    unsigned checks = 0, retired = 0;
    auto verify = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("Runtime continuity contract failed"); };
    phoenix_compat::RuntimeContinuity<Session> cache;
    using Owner = decltype(cache)::Owner;
    const Owner original{1,2};
    auto session = std::make_unique<Session>(retired);
    const auto* identity = session.get();
    session->swapchain_views_released = true;
    cache.park(original,std::move(session));
    verify(cache.size()==1 && retired==0);
    verify(!cache.take({1,3}) && !cache.take({2,2}));
    session=cache.take(original);
    verify(session.get()==identity && session->helper_pid==17 && session->completed_frames==42);
    verify(session->swapchain_views_released && cache.size()==0 && retired==0);
    for(unsigned reset=0;reset<1000;++reset) {
        cache.park(original,std::move(session));session=cache.take(original);
        verify(session.get()==identity && retired==0);
    }
    cache.park(original,std::move(session));cache.destroy_device(999);
    verify(cache.size()==1 && retired==0);
    cache.destroy_device(1);verify(cache.size()==0 && retired==1);
    cache.park({4,5},std::make_unique<Session>(retired));
    cache.prune([](auto owner){return owner.second!=5;});
    verify(cache.size()==0 && retired==2);
    for(uintptr_t owner=1;owner<=5;++owner) cache.park({owner,owner},std::make_unique<Session>(retired));
    verify(cache.size()==4 && !cache.take({1,1}) && retired==3);
    cache.clear();verify(cache.size()==0 && retired==7);
    cache.park({0,2},std::make_unique<Session>(retired));verify(cache.size()==0 && retired==8);
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"checks\":"<<checks<<"}\n";
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
