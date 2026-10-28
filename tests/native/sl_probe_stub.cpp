#include <windows.h>
#include "../../third_party/streamline/sl_core_types.h"
#if defined(_M_IX86)
#pragma comment(linker, "/EXPORT:slEvaluateFeature=_slEvaluateFeature")
#pragma comment(linker, "/EXPORT:slSetTagForFrame=_slSetTagForFrame")
#pragma comment(linker, "/EXPORT:slSetTag=_slSetTag")
#pragma comment(linker, "/EXPORT:ProbeCalls=_ProbeCalls")
#endif
static unsigned frame_calls, old_calls;
extern "C" __declspec(dllexport) sl::Result slEvaluateFeature(sl::Feature,
    const sl::FrameToken&, const sl::BaseStructure**, uint32_t, sl::CommandBuffer*) {
    return sl::Result::eErrorIO;
}
extern "C" __declspec(dllexport) sl::Result slSetTagForFrame(const sl::FrameToken&,
    const sl::ViewportHandle&, const sl::ResourceTag* tags, uint32_t count, sl::CommandBuffer*) {
    ++frame_calls;
    return count == 1 && tags && tags[0].resource && tags[0].resource->native == reinterpret_cast<void*>(1)
        ? sl::Result::eOk : sl::Result::eErrorIO;
}
extern "C" __declspec(dllexport) sl::Result slSetTag(const sl::ViewportHandle&,
    const sl::ResourceTag* tags, uint32_t count, sl::CommandBuffer*) {
    ++old_calls;
    return count == 1 && tags && tags[0].resource && tags[0].resource->native == reinterpret_cast<void*>(1)
        ? sl::Result::eOk : sl::Result::eErrorIO;
}
extern "C" __declspec(dllexport) unsigned ProbeCalls() { return frame_calls + old_calls; }
