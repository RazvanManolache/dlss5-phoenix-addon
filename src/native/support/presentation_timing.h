#pragma once
#include "../../../third_party/nlohmann/json.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace phoenix_addon::phoenix {
// Whole-game presentation windows, measured by the game process, never by a
// helper's render loop or by taking the reciprocal of a neural GPU interval.
class PresentationTiming {
    using Json = nlohmann::json;
  public:
    struct Sample { uint64_t frames, milliseconds; double fps; };
  private:
    struct Row { uint64_t frames=0, milliseconds=0, sequence=0; };
    std::map<std::string,Row> rows_;
    std::vector<float> history_;
    std::string context_;
    uint64_t revision_=0, tick_=0, frames_=0, sequence_=0;
    int mode_=-1;
    bool initialized_=false;
    static std::string key(const Json& row) {
        auto settings=row.at("settings");
        settings.erase("HelperProcess"); // Retained UI rows add this annotation later.
        return Json{{"mode",row.at("mode")},{"blocks",row.at("transformer_blocks")},
                    {"width",row.at("width")},{"height",row.at("height")},
                    {"settings",settings},{"presentation_scope",row.value("presentation_scope",std::string())}}.dump();
    }
  public:
    void clear() { rows_.clear();history_.clear();initialized_=false;frames_=0; }
    std::optional<Sample> present(uint64_t now, std::string_view context, uint64_t revision, int mode) {
        if (!initialized_ || context_!=context || revision_!=revision || mode_!=mode || now<tick_) {
            initialized_=true;context_=context;revision_=revision;mode_=mode;tick_=now;frames_=0;
            return std::nullopt; // Exclude windows spanning a workload switch.
        }
        ++frames_;
        if(now-tick_<1000)return std::nullopt;
        Sample sample{frames_,now-tick_,double(frames_)*1000./double(now-tick_)};
        tick_=now;frames_=0;
        if(history_.size()==120)history_.erase(history_.begin());
        history_.push_back(float(sample.fps));
        return sample;
    }
    void record(const Sample& sample, const Json& row) {
        if(!sample.milliseconds || !std::isfinite(sample.fps) || sample.fps<=0)return;
        const auto identity=key(row);
        if(!rows_.count(identity)&&rows_.size()>=128) {
            auto oldest=std::min_element(rows_.begin(),rows_.end(),[](const auto& a,const auto& b){return a.second.sequence<b.second.sequence;});
            rows_.erase(oldest);
        }
        auto& value=rows_[identity];value.frames+=sample.frames;value.milliseconds+=sample.milliseconds;value.sequence=++sequence_;
    }
    void record_active(const Sample& sample, const Json& report, int mode, unsigned blocks,
                       const std::string& scope = {}) {
        if(!report.is_object() || !report.contains("rows") || !report.contains("active_settings"))return;
        for(auto row:report.at("rows"))
            if(row.value("mode",-1)==mode && row.value("transformer_blocks",0u)==blocks &&
               row.at("settings")==report.at("active_settings")) {
                row["presentation_scope"]=scope;record(sample,row);break;
            }
    }
    std::optional<double> fps(const Json& row) const {
        const auto found=rows_.find(key(row));
        if(found==rows_.end() || !found->second.milliseconds)return std::nullopt;
        return double(found->second.frames)*1000./double(found->second.milliseconds);
    }
    const std::vector<float>& history() const {return history_;}
    Json report() const {
        Json rows=Json::array();
        for(const auto& [identity,row]:rows_)rows.push_back({{"workload",Json::parse(identity)},
            {"frames",row.frames},{"milliseconds",row.milliseconds},
            {"fps",row.milliseconds?double(row.frames)*1000./double(row.milliseconds):0.}});
        return {{"scope","Game presentation callbacks, including repeated frames; one-second windows exclude workload switches"},
                {"history",history_},{"rows",rows}};
    }
};
}
