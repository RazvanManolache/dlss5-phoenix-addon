// Correctly rounded FP32 division/sqrt for the normalization range. Vulkan's
// native approximations can cross the model's half/FP8 publication boundaries.
// Integer significands retain the CPU's rounding, including exact halfway ties.
#ifndef PHOENIX_IEEE_MATH_GLSL
#define PHOENIX_IEEE_MATH_GLSL
uint64_t phoenix_align_significand(uint64_t value,int shift) {
    if(shift>=0)return value<<uint(shift);
    uint right=uint(-shift);
    if(right>=64u)return uint64_t(value!=0ul);
    uint64_t mask=(1ul<<right)-1ul;
    return (value>>right)|uint64_t((value&mask)!=0ul);
}
int phoenix_high_bit(uint64_t value) {
    uint high=uint(value>>32u);
    return high!=0u?32+findMSB(high):findMSB(uint(value));
}
float phoenix_ieee_fma(float left,float right,float addend) {
    uint a=floatBitsToUint(left),b=floatBitsToUint(right),c=floatBitsToUint(addend);
    uint ea=(a>>23u)&255u,eb=(b>>23u)&255u,ec=(c>>23u)&255u;
    if(ea==255u||eb==255u||ec==255u)return fma(left,right,addend);
    uint ma=(a&0x7fffffu)|(ea!=0u?0x800000u:0u);
    uint mb=(b&0x7fffffu)|(eb!=0u?0x800000u:0u);
    uint mc=(c&0x7fffffu)|(ec!=0u?0x800000u:0u);
    if(ma==0u||mb==0u)return addend==0.0?uintBitsToFloat((a^b)&c&0x80000000u):addend;
    uint64_t product=uint64_t(ma)*uint64_t(mb);
    int ep=(ea!=0u?int(ea)-150:-149)+(eb!=0u?int(eb)-150:-149);
    int ce=ec!=0u?int(ec)-150:-149;
    int top=ep+phoenix_high_bit(product);
    if(mc!=0u)top=max(top,ce+findMSB(mc));
    uint64_t p=phoenix_align_significand(product,ep+55-top);
    uint64_t q=mc!=0u?phoenix_align_significand(uint64_t(mc),ce+55-top):0ul;
    uint sign=(a^b)&0x80000000u;
    uint64_t sum;
    if(((a^b^c)&0x80000000u)==0u)sum=p+q;
    else if(p>=q)sum=p-q;
    else {sum=q-p;sign=c&0x80000000u;}
    if(sum==0ul)return 0.0;
    int leading=phoenix_high_bit(sum),exponent=top+leading-55+127;
    if(exponent>=255)return uintBitsToFloat(sign|0x7f800000u);
    int dropped=exponent>0?leading-23:-top-94;
    if(dropped>=64)return uintBitsToFloat(sign);
    uint shift=uint(max(dropped,0));
    uint mantissa=dropped<0?uint(sum<<uint(-dropped)):uint(sum>>shift);
    if(shift!=0u){uint64_t low=sum&((1ul<<shift)-1ul),halfway=1ul<<(shift-1u);
        mantissa+=uint(low>halfway||(low==halfway&&(mantissa&1u)!=0u));}
    if(exponent<=0)return uintBitsToFloat(sign|mantissa);
    return uintBitsToFloat(sign|((uint(exponent)<<23u)+(mantissa-0x800000u)));
}
// Positive square-plus-accumulator in the normalization range. The mobile
// driver's native FP32 fma differs by one ULP for a real museum Q/K row. Half
// operands in GEMM do not expose that product rounding, but FP32 Q/K squares do.
float phoenix_square_add(float value,float accumulator) {
    uint a=floatBitsToUint(value)&0x7fffffffu,c=floatBitsToUint(accumulator);
    uint ea=a>>23u,ec=(c>>23u)&255u;
    if(ea==0u)return accumulator;
    if(ea==255u||ec==255u)return fma(value,value,accumulator);
    uint64_t product=uint64_t((a&0x7fffffu)|0x800000u)*uint64_t((a&0x7fffffu)|0x800000u);
    int ep=2*(int(ea)-150),etop=ep+phoenix_high_bit(product);
    int ce=int(ec)-150;
    int top=ec!=0u?max(etop,int(ec)-127):etop;
    uint64_t sum=phoenix_align_significand(product,ep+55-top);
    if(ec!=0u)sum+=phoenix_align_significand(uint64_t((c&0x7fffffu)|0x800000u),ce+55-top);
    int leading=phoenix_high_bit(sum),exponent=top+leading-55+127;
    if(exponent<=0||exponent>=255)return fma(value,value,accumulator);
    uint shift=uint(leading-23);
    uint q=uint(sum>>shift);
    uint64_t low=sum&((1ul<<shift)-1ul),halfway=1ul<<(shift-1u);
    q+=uint(low>halfway||(low==halfway&&(q&1u)!=0u));
    return uintBitsToFloat((uint(exponent)<<23u)+(q-0x800000u));
}
float phoenix_ieee_divide(float numerator, float denominator) {
    uint a=floatBitsToUint(numerator),b=floatBitsToUint(denominator);
    uint sign=(a^b)&0x80000000u,ea=(a>>23u)&255u,eb=(b>>23u)&255u;
    if((a&0x7fffffffu)==0u)return uintBitsToFloat(sign);
    if(ea==0u||eb==0u||ea==255u||eb==255u)return numerator/denominator;
    uint ma=(a&0x7fffffu)|0x800000u,mb=(b&0x7fffffu)|0x800000u;
    uint shift=ma<mb?24u:23u;
    int exponent=int(ea)-int(eb)+127-(ma<mb?1:0);
    if(exponent<=0||exponent>=254)return numerator/denominator;
    // Use restoring 32-bit division. Some mobile drivers miscompile uint64
    // division, even for 1.0/1.0. Remainders here always fit in 25 bits.
    uint q=uint(ma>=mb),r=ma-(ma>=mb?mb:0u);
    for(uint digit=0u;digit<shift;++digit) {
        r<<=1u;q<<=1u;
        if(r>=mb){r-=mb;q|=1u;}
    }
    q+=uint(r*2u>mb||(r*2u==mb&&(q&1u)!=0u));
    return uintBitsToFloat(sign|((uint(exponent)<<23u)+(q-0x800000u)));
}
float phoenix_ieee_sqrt(float value) {
    uint bits=floatBitsToUint(value),e=(bits>>23u)&255u;
    if(value<=0.0||e==0u||e==255u)return sqrt(value);
    int exponent=int(e)-127;
    uint64_t n=uint64_t((bits&0x7fffffu)|0x800000u)<<uint(23+(exponent&1));
    // Restoring integer square root. The radicand has at most 48 bits.
    uint64_t remainder=n,root=0ul,bit=1ul<<46u;
    while(bit!=0ul) {
        if(remainder>=root+bit){remainder-=root+bit;root=(root>>1u)+bit;}
        else root>>=1u;
        bit>>=2u;
    }
    root+=uint64_t(remainder>root);
    return uintBitsToFloat((uint((exponent>>1)+127)<<23u)+(uint(root)-0x800000u));
}
#endif
