#pragma once
#include "../../definition.h"

namespace phoenix_model::models {
// Portable decoder/graph and architecture-specific native RTX launch support.
inline constexpr Definition small_fp8{
    "small-fp8", "Lightweight FP8", "phoenix-cc-small-fp8/1", "WEIGHTS_CC_SMALL_FP8",
    "f16f2ef91fc207eb384f0db91a627766bd853fd8013b1f532d8ea6bd64c150fd",
    77, 302, 30, true, "",
    "2bc19bc09a7b89dd44f73e2ac0108fcc586e2154bdd0830f242f4f040e28789f",
    "f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962",
    {78, 18, 2, 11, {0, 0, 0, 0},
     {1, 3, 5, 7, 9, 18, 19, 59, 60, 69, 71, 73, 75, 77, 78}}, 2};
} // namespace phoenix_model::models
