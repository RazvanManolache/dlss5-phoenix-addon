#include "../motion_plugin.h"
#include <opencv2/video/tracking.hpp>
#include <memory>

extern "C" __declspec(dllexport) void* PhoenixMotionCreateV1() {
    try {
        return new cv::Ptr<cv::DISOpticalFlow>(cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_ULTRAFAST));
    } catch (...) { return nullptr; }
}

extern "C" __declspec(dllexport) void PhoenixMotionDestroyV1(void* opaque) {
    delete static_cast<cv::Ptr<cv::DISOpticalFlow>*>(opaque);
}

extern "C" __declspec(dllexport) bool PhoenixMotionRunV1(void* opaque, unsigned width, unsigned height,
    const unsigned char* current, unsigned pitch, const unsigned char* previous, float* output, unsigned output_pitch) {
    try {
        const cv::Mat a(height, width, CV_8UC1, const_cast<unsigned char*>(current), pitch);
        const cv::Mat b(height, width, CV_8UC1, const_cast<unsigned char*>(previous), width);
        cv::Mat flow(height, width, CV_32FC2, output, output_pitch);
        // Empty output disables DIS temporal seeding from unrelated upload contents.
        cv::Mat fresh;
        (*static_cast<cv::Ptr<cv::DISOpticalFlow>*>(opaque))->calc(a.isContinuous() ? a : a.clone(), b, fresh);
        fresh.copyTo(flow);
        return true;
    } catch (...) { return false; }
}
