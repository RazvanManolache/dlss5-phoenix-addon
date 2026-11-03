#include "../../src/integration/compatibility/capture_retirement.h"
#include <iostream>
#include <stdexcept>
int main()try {
 unsigned checks=0;auto check=[&](bool v){++checks;if(!v)throw std::runtime_error("Capture retirement regression");};
 unsigned polls=0,yields=0,clock=0;
 auto query=[&]{return ++polls==3?uint64_t(7):uint64_t(6);};
 auto expired=[&]{return ++clock>8;};auto yield=[&]{++yields;};
 check(phoenix_compat::poll_capture_retirement(7,6,true,query,expired,yield)==7);
 check(polls==3&&yields==3);
 polls=yields=clock=0;
 check(phoenix_compat::poll_capture_retirement(7,7,true,query,expired,yield)==7);
 check(polls==0&&yields==0&&clock==0);
 check(phoenix_compat::poll_capture_retirement(7,6,false,query,expired,yield)==6);
 check(polls==0&&yields==0&&clock==0);
 auto never=[&]{++polls;return uint64_t(6);};
 auto short_window=[&]{return ++clock>2;};
 check(phoenix_compat::poll_capture_retirement(7,6,true,never,short_window,yield)==6);
 check(polls==2&&yields==2);
 polls=yields=clock=0;
 auto lost=[&]{++polls;return UINT64_MAX;};
 check(phoenix_compat::poll_capture_retirement(7,6,true,lost,expired,yield)==UINT64_MAX);
 check(polls==1&&yields==1);
 std::cout<<"{\"passed\":true,\"cpu_only\":true,\"checks\":"<<checks<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what();return 1;}
