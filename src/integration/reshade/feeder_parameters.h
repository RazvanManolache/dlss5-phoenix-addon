#pragma once
#include "feeder_api.h"
#include <nvsdk_ngx.h>
#include <cstring>

namespace phoenix_addon {
// Internal adapter for the existing direct renderer. This is a local value
// container, not a feature handle or an NGX runtime call.
class FeederParameters final : public NVSDK_NGX_Parameter {
    const PhoenixFeederFrameV1& frame_;
  public:
    explicit FeederParameters(const PhoenixFeederFrameV1& frame) : frame_(frame) {}
#define PHOENIX_FEEDER_EMPTY(T) \
    void Set(const char*, T) override {} \
    NVSDK_NGX_Result Get(const char*, T*) const override { return NVSDK_NGX_Result_Fail; }
    PHOENIX_FEEDER_EMPTY(unsigned long long)
    PHOENIX_FEEDER_EMPTY(double)
    PHOENIX_FEEDER_EMPTY(ID3D11Resource*)
#undef PHOENIX_FEEDER_EMPTY
#define PHOENIX_FEEDER_SET(T) void Set(const char*, T) override {}
    PHOENIX_FEEDER_SET(float)
    PHOENIX_FEEDER_SET(unsigned)
    PHOENIX_FEEDER_SET(int)
    PHOENIX_FEEDER_SET(ID3D12Resource*)
    PHOENIX_FEEDER_SET(void*)
#undef PHOENIX_FEEDER_SET
    NVSDK_NGX_Result Get(const char* key, float* value) const override {
        if (!strcmp(key, NVSDK_NGX_Parameter_MV_Scale_X)) *value = frame_.motion_scale_x;
        else if (!strcmp(key, NVSDK_NGX_Parameter_MV_Scale_Y)) *value = frame_.motion_scale_y;
        else return NVSDK_NGX_Result_Fail;
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* key, unsigned* value) const override {
        if (!strcmp(key, NVSDK_NGX_Parameter_Reset)) *value = frame_.reset;
        else if (!strcmp(key, "DLSS.Feature.Create.Flags")) *value = frame_.depth_inverted ? 8u : 0u;
        else return NVSDK_NGX_Result_Fail;
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char* key, int* value) const override {
        unsigned result = 0;
        const auto status = Get(key, &result);
        if (NVSDK_NGX_SUCCEED(status)) *value = int(result);
        return status;
    }
    NVSDK_NGX_Result Get(const char* key, ID3D12Resource** value) const override {
        *value = !strcmp(key, NVSDK_NGX_Parameter_Output) ? frame_.output
               : !strcmp(key, NVSDK_NGX_Parameter_MotionVectors) ? frame_.motion
               : !strcmp(key, NVSDK_NGX_Parameter_Depth) ? frame_.depth : nullptr;
        return *value ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }
    NVSDK_NGX_Result Get(const char* key, void** value) const override {
        ID3D12Resource* resource = nullptr;
        const auto status = Get(key, &resource);
        *value = resource;
        return status;
    }
    void Reset() override {}
};
}
