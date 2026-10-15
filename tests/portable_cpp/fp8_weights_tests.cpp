#include "../../native_runtime/fp8_weights.h"
#include <cmath>
#include <cstdint>
#include <iostream>
// Independently interpret IEEE binary16 instead of reusing the encoder.
float half_value(uint16_t h) {
    unsigned e=(h>>10)&31,m=h&1023;
    float value=e?std::ldexp(1.0f+float(m)/1024,int(e)-15):std::ldexp(float(m),-24);
    return h&0x8000?-value:value;
}
int main() {
    for(unsigned code=0;code<256;++code){
        unsigned e=(code>>3)&15,m=code&7;
        float expected=e?std::ldexp(1.0f+float(m)/8,int(e)-7):float(m)/512;
        if(code&128)expected=-expected;
        float actual=half_value(phoenix_cpp::fp8_weight_half(uint8_t(code)));
        if(actual!=expected||std::signbit(actual)!=std::signbit(expected))return 1;
    }
    std::cout<<"All 256 legacy FP8 codes decode losslessly, including signed zero and +/-480.\n";
    return 0;
}
