#include "../../src/pipeline/session.h"
#include "../../src/pipeline/recovery.h"
#include "../../src/pipeline/screenshot_receipt.h"
#include <cassert>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    using namespace phoenix_pipeline;
    Options p;
    assert(p.valid());
    assert(std::abs(p.normalized_flow_cost()-.1f)<1e-6f);
    p.flow_cost=500;assert(std::abs(p.normalized_flow_cost()-.5f)<1e-6f);p={};
    p.motion_enabled = true;
    assert(choose_motion(p, true, true).source == MotionSource::Engine);
    assert(choose_motion(p, false, true).source == MotionSource::Nvidia);
    assert(choose_motion(p, false, false).source == MotionSource::Zero);
    p.motion_source = int(MotionSource::Engine);
    assert(choose_motion(p, false, true).source == MotionSource::Zero);
    assert((flow_extent(1920, 1080, 180) == std::array<unsigned, 2>{320, 180}));
    assert((flow_extent(1080, 1920, 180) == std::array<unsigned, 2>{101, 180}));
    p.order = {0, 0, 1, 2}; assert(!p.valid()); p = {};
    p.latency_ms = std::numeric_limits<float>::quiet_NaN(); assert(!p.valid()); p = {};
    Monitor monitor; monitor.configure(p);
    assert(!monitor.enabled() && !monitor.gpu_sample(0));
    monitor.complete({}, -1); assert(!monitor.report().completed);
    p.monitoring = 2; p.sample_interval = 30; monitor.configure(p);
    assert(monitor.gpu_sample(30) && !monitor.gpu_sample(29));
    const double times[]{1, 2, 3, 4, 5}; monitor.complete(times, 25);
    assert(monitor.report().completed == 1 && monitor.report().frame_age_ms == 25);
    p.priority = 1; p.latency_ms = 20; LatencyPolicy policy;
    for (int i = 0; i < 29; ++i) assert(policy.sample(p, 50, false, {true,true,true,true}) == BudgetKnob::None);
    assert(policy.sample(p, 50, false, {true,true,true,true}) == BudgetKnob::Flow);
    assert(policy.sample(p, 50, true, {true,true,true,true}) == BudgetKnob::None);
    for (int i = 0; i < 60; ++i) assert(policy.sample(p, 50, false, {true,true,true,true}) == BudgetKnob::None);
    policy.reset();
    for(int i=0;i<119;++i)assert(policy.sample(p,5,false,{false,false,false,false,true})==BudgetKnob::None);
    assert(policy.sample(p,5,false,{false,false,false,false,true})==BudgetKnob::RestoreFlow);
    Discontinuity detector; FrameIdentity frame{1, 1, 1, 1920, 1080};
    std::vector<uint8_t> a(64 * 36), b(a.size()); std::mt19937 random(9341);
    for (auto& x : a) x = uint8_t(random()); for (auto& x : b) x = uint8_t(random());
    assert(detector.observe(frame, false, true, a, 64, 36) == ResetReason::FirstFrame);
    ++frame.sequence; ++frame.timestamp_ns;
    assert(detector.observe(frame, false, true, a, 64, 36) == ResetReason::None);
    ++frame.sequence; ++frame.timestamp_ns;
    assert(detector.observe(frame, false, true, b, 64, 36) == ResetReason::Cut);
    ++frame.stream; ++frame.sequence;
    assert(detector.observe(frame, false, false) == ResetReason::SourceChanged);
    frame.timestamp_ns -= 1;
    assert(detector.observe(frame, false, false) == ResetReason::Timestamp);
    detector.clear();frame={1,1,INT64_MIN,31,19};detector.observe(frame,false,false);
    frame.sequence=2;frame.timestamp_ns=INT64_MAX;
    assert(detector.observe(frame,false,false)==ResetReason::Timestamp);
    detector.clear();frame={1,1,0,31,19};detector.observe(frame,false,false);
    frame.sequence=3;frame.timestamp_ns=1;assert(detector.observe(frame,false,false)==ResetReason::Timestamp);
    detector.clear();frame={1,1,0,31,19};std::vector<uint8_t> tiny_gray(4,100);
    detector.observe(frame,false,true,tiny_gray,2,2);++frame.sequence;++frame.timestamp_ns;
    assert(detector.observe(frame,false,true,tiny_gray,2,2)==ResetReason::None);
    assert(argc == 2);
    const auto image=std::filesystem::path(argv[1])/"saved-image.png";
    {std::ofstream file(image,std::ios::binary);file<<"receipt fixture";}
    auto unrelated=image;unrelated+=L".phoenix.json";
    {std::ofstream file(unrelated);file<<"preserve user file";}
    const std::string image_hash(64,'a');
    const auto receipt=save_screenshot_receipt(image,image_hash,{{"frame",42},{"schema","caller cannot override"}});
    assert(receipt!=unrelated);
    {std::ifstream file(unrelated);std::string text;std::getline(file,text);assert(text=="preserve user file");}
    {std::ifstream file(receipt);const auto facts=Json::parse(file);
     assert(facts.at("schema")=="phoenix-screenshot/1"&&facts.at("frame")==42);
     assert(facts.at("image").at("sha256")==image_hash&&facts.at("image").at("bytes")==15);}
    assert(save_screenshot_receipt(image,image_hash,{{"frame",43}})!=receipt);
    bool refused=false;try{save_screenshot_receipt(image,"invalid",{});}catch(const std::invalid_argument&){refused=true;}
    assert(refused);
    const auto path = std::filesystem::path(argv[1]) / "journal.json";
    const Json original{{"model", "original"}}, lightweight{{"model", "small-fp8"}};
    { RecoveryJournal journal(path, original); journal.begin(lightweight);
      assert(journal.pending()); journal.completed_frame(); }
    { RecoveryJournal journal(path, original); assert(journal.recover_interrupted() == original);
      assert(!journal.pending()); journal.begin(lightweight);
      for (unsigned i = 0; i < 29; ++i) assert(!journal.completed_frame());
      assert(journal.completed_frame()); assert(journal.working() == lightweight); }
    { RecoveryJournal journal(path, original); assert(!journal.recover_interrupted());
      journal.begin(lightweight);assert(!journal.pending());
      journal.begin(original); assert(journal.failure("provider initialization failed") == lightweight); }
    { std::ifstream in(path);Json saved;in>>saved;in.close();saved["requested"]=original;
      {std::ofstream out(path);out<<saved.dump();}
      RecoveryJournal journal(path,lightweight);assert(journal.take_requested()==original);assert(!journal.take_requested());
      assert(journal.working()==lightweight);journal.begin(original);assert(journal.failure("requested provider failed")==lightweight); }
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"motion_priority\":true,\"scene_cuts\":true,\"monitoring_off\":true,\"startup_rollback\":true}\n";
}
