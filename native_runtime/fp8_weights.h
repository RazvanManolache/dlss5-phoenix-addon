#pragma once
#include <cstdint>
namespace phoenix_cpp {
// Lossless legacy E4M3 -> IEEE binary16. Unlike hardware E4M3FN, codes
// 0x7f/0xff are finite +/-480 in this model's existing scalar decoder.
inline uint16_t fp8_weight_half(uint8_t code) {
    unsigned sign=(unsigned(code)&128u)<<8u,e=(code>>3u)&15u,m=code&7u;
    if(e)return uint16_t(sign|((e+8u)<<10u)|(m<<7u));
    if(!m)return uint16_t(sign);
    unsigned leading=m>=4u?2u:m>=2u?1u:0u;
    return uint16_t(sign|((leading+6u)<<10u)|((m-(1u<<leading))<<(10u-leading)));
}
}
