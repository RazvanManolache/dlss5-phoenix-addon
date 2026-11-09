#include "pixels.h"
#include <array>
#include <cstdio>
#include <limits>

int main() {
    for (bool bgra : {false,true}) {
        // Every 8-bit channel value must survive a no-op inference round trip,
        // with encoded values unchanged and arbitrary alpha preserved.
        for (int v=0;v<256;++v) {
            std::array<uint8_t,8> bytes{uint8_t(v),uint8_t(255-v),128,17,23,45,67,89};
            auto original=bytes;
            float rgb[6]{};
            phoenix_linux::unpack(bytes.data(),rgb,2,bgra);
            phoenix_linux::pack(rgb,bytes.data(),2,bgra);
            if (bytes!=original) { std::fprintf(stderr,"Round trip failed at %d\n",v); return 1; }
        }
        float rgb[]{-1.f,2.f,std::numeric_limits<float>::quiet_NaN()};
        uint8_t bytes[]{1,2,3,99};
        phoenix_linux::pack(rgb,bytes,1,bgra);
        if (bytes[bgra?2:0]!=0 || bytes[1]!=255 || bytes[bgra?0:2]!=0 || bytes[3]!=99) return 2;
    }
    // Compare both sides of every quantization threshold with the established
    // lround behavior. Adjacent float32 values catch add-0.5 shortcuts.
    for (int code=0;code<255;++code) {
        const float center=(float(code)+0.5f)/255.f;
        for (float value : {std::nextafter(center,0.f),center,std::nextafter(center,1.f)}) {
            float rgb[]{value,value,value};
            uint8_t bytes[]{0,0,0,137};
            phoenix_linux::pack(rgb,bytes,1,false);
            const auto expected=uint8_t(std::lround(value*255.f));
            if (bytes[0]!=expected || bytes[1]!=expected || bytes[2]!=expected || bytes[3]!=137)
                return 3;
        }
    }

}
