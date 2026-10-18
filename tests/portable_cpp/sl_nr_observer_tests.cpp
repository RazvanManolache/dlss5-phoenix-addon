#include "../../src/native/support/sl_shim.h"
#include <iostream>
#include <stdexcept>
using namespace phoenix_addon::sl_shim;
static unsigned checks = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        SlObserver observer;
        SlNRInputs result;
        SlTag left[] = {{100, kTagUpliftInputColor, 640, 360},
                        {101, kTagMotion, 640, 360},
                        {102, kTagUpliftControlMask, 640, 360}};
        SlTag right[] = {{200, kTagUpliftInputColor, 640, 360},
                         {201, kTagMotion, 640, 360},
                         {202, kTagUIAlpha, 640, 360}};
        observer.note_tags(10, left, 3, 0);
        observer.note_tags(10, right, 3, 1);
        check(observer.nr_inputs(10, 0, result) && result.color.resource_key == 100 &&
                  result.control_mask.resource_key == 102 && !result.ui_alpha.resource_key,
              "left viewport resources mixed");
        check(observer.nr_inputs(10, 1, result) && result.color.resource_key == 200 &&
                  result.ui_alpha.resource_key == 202 && !result.control_mask.resource_key,
              "right viewport resources mixed");
        check(!observer.nr_inputs(11, 0, result) && !result.color.resource_key, "stale frame reused");
        check(!observer.nr_inputs(10, 2, result), "unknown viewport reused");
        SlTag retired{0, kTagUpliftControlMask};
        observer.note_tags(10, &retired, 1, 0);
        check(observer.nr_inputs(10, 0, result) && !result.control_mask.resource_key, "null-tag retirement ignored");
        retired.buffer_type = kTagMotion;
        observer.note_tags(10, &retired, 1, 0);
        check(!observer.nr_inputs(10, 0, result), "retired motion reused");
        observer.note_tags_global(left, 3);
        check(!observer.nr_inputs(11, 0, result), "global pointer reused for exact frame");
        observer.note_evaluate(kFeatureDLSSNR, 10, 640, 360);
        check(observer.find(kFeatureDLSSNR)->color_w == 640, "NR input tag unrecognized");
        check(!observer.drive_template(), "unvalidated NR provider enabled by observer");
        SlTag rr[] = {{300, kTagScalingInputColor, 320, 180}, {301, kTagMotion, 320, 180}};
        observer.note_tags(12, rr, 2, 0);
        observer.note_evaluate(kFeatureDLSSNR, 12, 640, 360);
        check(!observer.find(kFeatureDLSSNR)->color_w, "RR color substituted for missing NR tag");
        observer.note_evaluate(kFeatureDLSSRR, 12, 640, 360);
        check(observer.drive_template() && observer.drive_template()->color_w == 320, "RR template regressed");
        for (unsigned frame = 13; frame < 24; ++frame) observer.note_tags(frame, left, 3, 0);
        check(observer.frames_held() == 8 && !observer.nr_inputs(10, 1, result), "unbounded or expired frame retention");
        std::cout << checks << " Streamline NR observation checks passed; no provider/device execution\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
