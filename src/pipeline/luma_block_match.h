#pragma once

namespace phoenix_pipeline {
// One estimate per 8x8 luma block. Keep work bounded with a two-pixel
// search and then refine in one-pixel steps around the winning offset.
inline constexpr char luma_block_match[] = R"hlsl(
Texture2D<float> Current : register(t0);
Texture2D<float> Previous : register(t1);
RWTexture2D<int2> Flow : register(u0);
cbuffer C : register(b0) { uint W, H; uint unused[7]; };

float cost(int2 center, int2 delta, int2 edge, float patch[16]) {

    float error = 0;
    [unroll] for (int y = 0; y < 4; ++y)
        [unroll] for (int x = 0; x < 4; ++x) {
            int2 pos = center + delta + int2(x, y) * 2 - 3;
            error += abs(patch[y * 4 + x] - Previous.Load(int3(clamp(pos, 0, edge), 0)));
        }
    bool outside = any(center + delta - 3 < 0) || any(center + delta + 3 > edge);
    return outside ? 2.0 : error * (1.0 / 16.0);
}

[numthreads(16,16,1)]
void main(uint3 thread : SV_DispatchThreadID) {
    if (thread.x >= W || thread.y >= H) return;
    uint w,h;
    Current.GetDimensions(w,h);
    int2 edge = int2(w - 1, h - 1);
    int2 center = int2(thread.xy) * 8 + 4;
    const int2 invalid = int2(16384, 16384);
    if (any(center - 3 < 0) || any(center + 3 > edge)) {
        Flow[thread.xy] = invalid;
        return;
    }
    float patch[16];
    float low = 1, high = 0;
    [unroll] for (int y = 0; y < 4; ++y)
        [unroll] for (int x = 0; x < 4; ++x) {
            float luma = Current.Load(int3(center + int2(x,y)*2 - 3,0));
            patch[y*4+x] = luma;
            low = min(low,luma);
            high = max(high,luma);
        }
    float zero = cost(center, int2(0,0), edge, patch);
    if (zero < 0.004) {
        Flow[thread.xy] = int2(0,0);
        return;
    }
    float best = zero, runner_up = 2;
    int2 winner = int2(0,0);
    int radius = H >= 64 ? 12 : 8;
    [loop] for (int coarse_y = -radius; coarse_y <= radius; coarse_y += 2)
        [loop] for (int coarse_x = -radius; coarse_x <= radius; coarse_x += 2) {
            if (coarse_x == 0 && coarse_y == 0) continue;
            int2 offset = int2(coarse_x,coarse_y);
            float error = cost(center,offset,edge,patch);
            if (error < best) {
                runner_up = best;
                best = error;
                winner = offset;
            } else runner_up = min(runner_up,error);
        }
    const int2 coarse = winner;
    [unroll] for (int refine_y = -1; refine_y <= 1; ++refine_y)
        [unroll] for (int refine_x = -1; refine_x <= 1; ++refine_x) {
            if (refine_x == 0 && refine_y == 0) continue;
            int2 offset = coarse + int2(refine_x,refine_y);
            float error = cost(center,offset,edge,patch);
            if (error < best) {
                best = error;
                winner = offset;
            }
        }
    bool reliable = high-low >= 0.025 && best <= 0.075 &&
                    runner_up-best >= max(0.0015, best*0.05);
    Flow[thread.xy] = reliable ? winner : invalid;
})hlsl";
}
