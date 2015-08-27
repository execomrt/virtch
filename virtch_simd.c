/*	

Copyright 2015 realtech VR 

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

*/

/*==============================================================================
 
 $Id$
 
 SIMD mixing / conversion functions. S.Denis (realtech@users.sf.net) 04/2015
 
 virtch_downmix_32_16:
 SSE: 25542 cycles vs 76294 (298% faster)
 ARM: 512 cycles vs 3865
 AVX2: Not tested
 
 virtch_mix_stereo test...
 SSE: 64810 cycles vs 122887 (189% faster)
 NEON: 4076 cycles vs 9422
 AVX2: Not tested
 
 ==============================================================================*/

#include "virtch_simd.h"

#ifdef __ALTIVEC__
#ifdef __GNUC__
#include <ppc_intrinsics.h>
#endif
typedef vector float v4sf;  // vector of 4 float
typedef vector signed int v4si;  // vector of 4 uint32
#elif defined __ARM_NEON__
#include <arm_neon.h>
typedef float32x4_t v4sf;  // vector of 4 float
typedef int32x4_t v4si;  // vector of 4 uint32
typedef int16x8_t v8sw;  // vector of 4 uint32

#ifdef _WPHONE
inline int16x4_t MAKE_16x4(int16_t a, int16_t b, int16_t c, int16_t d) 
{ int16x4_t ret; 
ret.n64_i16[0] = a;
ret.n64_i16[1] = b; 
ret.n64_i16[2] = c; 
ret.n64_i16[3] = d;
return ret; }
#else
#define MAKE_16x4(a,b,c,d) {a, b, c, d}
#endif

#elif VMIX_SIMD == VMIX_SIMD_SSE ||  VMIX_SIMD == VMIX_SIMD_AVX ||  VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512
#ifdef _MSC_VER
#include <intrin.h>
#include <immintrin.h>
#define FORCE_INLINE(type) static __inline type __attribute__((__always_inline__, __nodebug__))
#else
#include <immintrin.h>
#define FORCE_INLINE(type) static __inline type
#endif

typedef __m128 v4sf;  // vector of 4 float
typedef __m128i v4si;  // vector of 4 uint32

#ifdef __AVX__
typedef __m256 v8sf;  // vector of 8 uint32
typedef __m256i v8si;  // vector of 8 float
#endif

#ifdef __AVX512F__
typedef __m512 v16sf;  // vector of 16 float
typedef __m512i v16si;  // vector of 16 uint32
#endif

#endif

// Allow higher quality
// #define VIRTCH_HQ


#ifdef VIRTCH_HQ

FORCE_INLINE(sample_t) GetLerpSample(const sample_t* const srce, size_t idx)
{
    size_t i = idx>>FRACBITS;
    streamsample_t f = idx&FRACMASK;
    return (sample_t)(((((streamsample_t)srce[i+0] * (FRACMASK+1L-f)) +
                        ((streamsample_t)srce[i+1] * f)) >> FRACBITS));
}

#define FETCH_SAMPLE(src, offset, increment) GetLerpSample(src, offset); offset += increment

#else

#define FETCH_SAMPLE(src, offset, increment) ((sample_t)src[(offset) >> FRACBITS]); offset += increment

#endif

static int virtch_features = 0;

void virtch_set_features(int features)
{
    virtch_features = features;
}

// Convert 32 to 16
#define BITSHIFT_VOL_32_16 (BITSHIFT )

// Convert 32 to 8
#define BITSHIFT_VOL_32_8  (BITSHIFT + 8)

// Clamp
#define CLAMP_F(x) ((x) < -1.0f ? -1.0f : ((x) > 1.0f ? 1.0f : 0.0f))

// Not yet optimized
void virtch_downmix_32_8(const streamsample_t* src,
                         int8_t* dst,
                         size_t n)
{
    
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127);
            src++;
            n--;
            if (!n) break;
        }
    
    
    
    
    while(n>0) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127); src++;
        n--;
    }
    
}

// Not yet optimized
/* shifting fudge factor for FP scaling, should be 0 < FP_SHIFT < BITSHIFT */
#define FP_SHIFT 4

void virtch_downmix_32_fp32(const streamsample_t* src,
                            float* dst,
                            size_t n)
{
    const float k = ((1.0f / 32768.0f) / (1 <<FP_SHIFT));
    
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            float sample = (*src++ >> (BITSHIFT-FP_SHIFT)) * k;
            *dst++ = CLAMP_F(sample);
            n--;
            if (!n)
            {
                return;
            }
        }
    
    
    
    
    
    while(n>0) {
        float sample = (*src++ >> (BITSHIFT-FP_SHIFT)) * k;
        *dst++ = CLAMP_F(sample);
        
        n--;
    }
    
}


void virtch_downmix_32_16(const streamsample_t* src,
                          int16_t* dst, // dst is REQUIRED to be aligned
                          size_t n) // n = number of sample (double this value for stereo)
{
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_16, 32767);
            src++;
            n--;
            if (!n)
            {
                return;
            }
        }
    
#if VMIX_SIMD == VMIX_SIMD_NEON
    int remain = n&7;
    if (virtch_features)
    {
        // Verified
        for (n>>=3;n; n--)
        {
            v4si v0 = vld1q_s32((int32_t*)(src+0));
            v4si v1 = vld1q_s32((int32_t*)(src+4));
            int16x4_t v2 = vqshrn_n_s32(v0, BITSHIFT_VOL_32_16);
            int16x4_t v3 = vqshrn_n_s32(v1, BITSHIFT_VOL_32_16);
            // Vector saturating narrow integer
            v4si v5 = vcombine_s16 (v2, v3);
            vst1q_s32((int32_t*)(dst+0), v5);
            
            dst+=8;
            src+=8;
        }
        n = remain;
        
    }
    
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
    int remain = n&7;
    if (virtch_features)
    {
        // Untested, looks correct
        for (n>>=3;n; n--)
        {
            v4si v0 = vec_ld(0, (v4si*)(src+0));
            v4si v1 = vec_ld(0, (v4si*)(src+4));
            v4si v2 = vec_sra(v0, vec_splat_u32(BITSHIFT_VOL_32_16));
            v4si v3 = vec_sra(v1, vec_splat_u32(BITSHIFT_VOL_32_16));
            // Vector saturating narrow integer
            v4si v5 = vec_packs(v1, v2);
            vec_st(v5, 0, dst);
            
            dst+=8;
            src+=8;
        }
        n = remain;
    }
    
    
#elif VMIX_SIMD == VMIX_SIMD_AVX512
    int remain = n&31;
    if (virtch_features)
    {
        // Untested, no hw access
        for (n>>=5;n; n--)
        {
            v16si v0 = _mm512_load_si512((v16si*)(src+0));
            v16si v1 = _mm512_load_si512((v16si*)(src+16));
            v16si v2 = _mm512_srai_epi32(v0, BITSHIFT_VOL_32_16);
            v16si v3 = _mm512_srai_epi32(v1, BITSHIFT_VOL_32_16);
            // AVX512f: Packs the 32 signed 32-bit integers from a and b into signed 16-bit integers and saturates
            v16si v5 = _mm512_packs_epi32(v2, v3);
            _mm512_store_si512((v8si*)(dst+0), v5);
            dst+=32;
            src+=32;
        }
        n = remain;
    }
    
    
#elif VMIX_SIMD == VMIX_SIMD_AVX2
    int remain = n&15;
    if (virtch_features)
    {
        // Untested, no hw access
        for (n>>=4;n; n--)
        {
            v8si v0 = _mm256_load_si256((v8si*)(src+0));
            v8si v1 = _mm256_load_si256((v8si*)(src+8));
            v8si v2 = _mm256_srai_epi32(v0, BITSHIFT_VOL_32_16);
            v8si v3 = _mm256_srai_epi32(v1, BITSHIFT_VOL_32_16);
            // AVX2: Packs the 16 signed 32-bit integers from a and b into signed 16-bit integers and saturates
            v8si v5 = _mm256_packs_epi32(v2, v3);
            _mm256_store_si256((v8si*)(dst+0), v5);
            dst+=16;
            src+=16;
        }
        n = remain;
    }
    
    
#elif VMIX_SIMD == VMIX_SIMD_SSE
    
    int remain = n&7;
    if (virtch_features)
    {
        // Verified
        for (n>>=3;n; n--)
        {
            v4si v0 = _mm_load_si128((v4si*)(src+0));
            v4si v1 = _mm_load_si128((v4si*)(src+4));
            // Shifts the 4 signed or unsigned 32-bit integers in a right by count bits while shifting in zeros.
            v4si v2 = _mm_srai_epi32(v0, BITSHIFT_VOL_32_16);
            v4si v3 = _mm_srai_epi32(v1, BITSHIFT_VOL_32_16);
            // Packs the 8 signed 32-bit integers from a and b into signed 16-bit integers and saturates
            v4si v5 = _mm_packs_epi32(v2, v3);
            _mm_store_si128((v4si*)(dst+0), v5);
            dst+=8;
            src+=8;
        }
        n = remain;
    }
    
    
#endif
    
    while(n>0) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_16, 32767);
        src++;
        n--;
    }
}

// mix dst, with 16bit sample.
size_t virtch_mix_mono(const sample_t* src,
                       const int32_t* vol,
                       streamsample_t* dst,
                       size_t offset,
                       size_t increment,
                       size_t n)
{
    sample_t sample;
    
    
    while(n--) {
        sample = FETCH_SAMPLE(src, offset, increment);
        *dst++ += vol[0] * sample;
    }
    return offset;
}

size_t virtch_mix_stereo(const sample_t* src,
                         const int32_t * vol,
                         streamsample_t* dst,
                         size_t offset,
                         size_t increment,
                         size_t n)
{
    sample_t sample;
    size_t remain = n;
    
    // dst can be misaligned ...
    if (virtch_features && n >= 4)
        while(!SAMPLE_ALIGNED(dst, 4)) {
            sample = FETCH_SAMPLE(src, offset, increment);
            *dst++ += vol[0] * sample;
            *dst++ += vol[1] * sample;
            n--;
            if (n == 0)
                break;
        }
    
    // src is always aligned ...
    
#if VMIX_SIMD == VMIX_SIMD_AVX2
    
    if (virtch_features)
    {
        // Untested, need hw access
        v8si v0 = _mm256_set_epi16(0, vol[1],
                                   0, vol[0],
                                   0, vol[1],
                                   0, vol[0],
                                   0, vol[1],
                                   0, vol[0],
                                   0, vol[1],
                                   0, vol[0]
                                   );
        remain = n&7;
        for (n>>=3;n; n--)
        {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            v8si v1 = _mm256_set_epi16(0, s3, 0, s3, 0, s2, 0, s2, 0, s1, 0, s1, 0, s0, 0, s0);
            v8si v2 = _mm256_set_epi16(0, s7, 0, s7, 0, s6, 0, s6, 0, s5, 0, s5, 0, s4, 0, s4);
            v8si v3 = _mm256_load_si256((v8si*)(dst+0));
            v8si v4 = _mm256_load_si256((v8si*)(dst+8));
            _mm256_store_si256((v8si*)(dst+0), _mm256_add_epi32(v3, _mm256_madd_epi16(v0, v1)));
            _mm256_store_si256((v8si*)(dst+8), _mm256_add_epi32(v4, _mm256_madd_epi16(v0, v2)));
            dst+=16;
        }
    }
    
    
#elif VMIX_SIMD == VMIX_SIMD_NEON
    
    if (virtch_features)
    {
        // Verified
        // Unrolled 8 samples gave 10% speedup
        int16x4_t v0 = MAKE_16x4((int16_t)vol[0], (int16_t)vol[1], (int16_t)vol[0], (int16_t)vol[1] );
        remain = n&7;
        for (n>>=3;n; n--)
        {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            
            
			int16x4_t v1 = MAKE_16x4( s0, s0, s1, s1 );
			int16x4_t v2 = MAKE_16x4( s2, s2, s3, s3 );
			int16x4_t v5 = MAKE_16x4( s4, s4, s5, s5 );
			int16x4_t v6 = MAKE_16x4( s6, s6, s7, s7 );
            
            
            int32x4_t v3 = vld1q_s32((int32_t*)(dst+0)); // src: a0, a1, a2, a3
            int32x4_t v4 = vld1q_s32((int32_t*)(dst+4));
            int32x4_t v7 = vld1q_s32((int32_t*)(dst+8)); // src: a0, a1, a2, a3
            int32x4_t v8 = vld1q_s32((int32_t*)(dst+12));
            
            vst1q_s32((int32_t*)(dst+0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+4), vmlal_s16(v4, v0, v2)); // vmlal_lane_s16 ?
            vst1q_s32((int32_t*)(dst+8), vmlal_s16(v7, v0, v5)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+12), vmlal_s16(v8, v0, v6)); // vmlal_lane_s16 ?
            dst+=16;
        }
    }
    
    
#elif VMIX_SIMD == VMIX_SIMD_SSE
    
    if (virtch_features)
    {
        // Verified
        v4si v0 = _mm_set_epi16(0, vol[1],
                                0, vol[0],
                                0, vol[1],
                                0, vol[0]);
        remain = n & 7;
        for (n>>=3;n; n--)
        {
            // Mono sample
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            
            v4si v1 = _mm_set_epi16(0, s1, 0, s1, 0, s0, 0, s0);
            v4si v2 = _mm_set_epi16(0, s3, 0, s3, 0, s2, 0, s2);
            v4si v5 = _mm_set_epi16(0, s5, 0, s5, 0, s4, 0, s4);
            v4si v6 = _mm_set_epi16(0, s7, 0, s7, 0, s6, 0, s6);
            
            v4si v3 = _mm_load_si128((v4si*) (dst + 0));
            v4si v4 = _mm_load_si128((v4si*) (dst + 4));
            v4si v7 = _mm_load_si128((v4si*) (dst + 8));
            v4si v8 = _mm_load_si128((v4si*) (dst + 12));
            
            // Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
            _mm_store_si128((v4si*)(dst+0), _mm_add_epi32(v3, _mm_madd_epi16(v0, v1))); // r0 := (a0 * b0) + (a1 * b1)
            _mm_store_si128((v4si*)(dst+4), _mm_add_epi32(v4, _mm_madd_epi16(v0, v2)));
            _mm_store_si128((v4si*)(dst+8), _mm_add_epi32(v7, _mm_madd_epi16(v0, v5)));
            _mm_store_si128((v4si*)(dst+12), _mm_add_epi32(v8, _mm_madd_epi16(v0, v6)));
            
            dst+=16;
        }
    }
    
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
    
    if (virtch_features)
    {
        // Untested, from virtch.c
        int32_t volq[] = {vol[0], vol[1], vol[0], vol[1]};
        vector signed short r0 = vec_ld(0, volq);
        vector signed short v0 = vec_perm(r0, r0, (vector unsigned char)(0, 1, // l
                                                                         0, 1, // l
                                                                         2, 3, // r
                                                                         2, 1, // r
                                                                         0, 1, // l
                                                                         0, 1, // l
                                                                         2, 3, // r
                                                                         2, 3  // r
                                                                         ));
        sample_t s[8];
        remain = n&3;
        for(n>>=2;n; n--)
        {
            // Load constants
            s[0] = FETCH_SAMPLE(src, offset, increment);
            s[1] = FETCH_SAMPLE(src, offset, increment);
            s[2] = FETCH_SAMPLE(src, offset, increment);
            s[3] = FETCH_SAMPLE(src, offset, increment);
            s[4] = 0;
            
            vector short int r1 = vec_ld(0, s);
            vector signed short v1 = vec_perm(r1, r1, (vector unsigned char)(0*2, 0*2+1, // s0
                                                                             4*2, 4*2+1, // 0
                                                                             0*2, 0*2+1, // s0
                                                                             4*2, 4*2+1, // 0
                                                                             1*2, 1*2+1, // s1
                                                                             4*2, 4*2+1, // 0
                                                                             1*2, 1*2+1, // s1
                                                                             4*2, 4*2+1  // 0
                                                                             ));
            
            vector signed short v2 = vec_perm(r1, r1, (vector unsigned char)(2*2, 2*2+1, // s2
                                                                             4*2, 4*2+1, // 0
                                                                             2*2, 2*2+1, // s2
                                                                             4*2, 4*2+1, // 0
                                                                             3*2, 3*2+1, // s3
                                                                             4*2, 4*2+1, // 0
                                                                             3*2, 3*2+1, // s3
                                                                             4*2, 4*2+1  // 0
                                                                             ));
            vector signed int v3 = vec_ld(0, dst);
            vector signed int v4 = vec_ld(0, dst + 4);
            vector signed int v5 = vec_mule(v0, v1);
            vector signed int v6 = vec_mule(v0, v2);
            
            vec_st(vec_add(v3, v5), 0, dst);
            vec_st(vec_add(v4, v6), 0x10, dst);
            
            dst+=8;
        }
    }
#endif
    
    // Remaining bits ...
    while(remain--) {
        sample = FETCH_SAMPLE(src, offset, increment);
        
        dst[0] += vol[0] * sample;
        dst[1] += vol[1] * sample;
        dst+= 2;
    }
    return offset;
}

// Stereo samples
size_t virtch_mix_stereo_st(const sample_t* src,
                            const int32_t * vol,
                            streamsample_t* dst,
                            size_t offset,
                            size_t increment,
                            size_t n)
{
    
    if (virtch_features && n > 0)
    {
        while (!SAMPLE_ALIGNED(dst, 4)) {
            dst[0] += vol[0] * (streamsample_t)FETCH_SAMPLE(src, offset, increment);
            dst[1] += vol[1] * (streamsample_t)FETCH_SAMPLE(src, offset, increment);
            dst+=2;
            
            n--;
            if (!n)
                break;
        }
    }
   	
    
    
#if VMIX_SIMD == VMIX_SIMD_NEON
    if (virtch_features)
    {
        // Verified
		int16x4_t v0 = MAKE_16x4((int16_t) vol[0], (int16_t) vol[1], (int16_t) vol[0], (int16_t) vol[1] );
        int remain = n&3;
        for (n>>=2;n; n--)
        {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            
            int16x4_t v1 = MAKE_16x4( s0, s1, s2, s3 );
			int16x4_t v4 = MAKE_16x4(s4, s5, s6, s7);
            
            int32x4_t v3 = vld1q_s32((int32_t*)(dst+0)); // src: a0, a1, a2, a3
            int32x4_t v5 = vld1q_s32((int32_t*)(dst+4)); // src: a0, a1, a2, a3
            
            vst1q_s32((int32_t*)(dst+0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+4), vmlal_s16(v5, v0, v4)); // v3+= v0 * v1
            
            
            dst+=8;
        }
        n = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_SSE
    
    if (virtch_features)
    {
        // Verified
        v4si v0 = _mm_set_epi16(0, vol[1],
                                0, vol[0],
                                0, vol[1],
                                0, vol[0]);
        int remain = n&1;
        for (n>>=1;n; n--)
        {
            // Mono sample
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            v4si v1 = _mm_set_epi16(0, s3, 0, s2, 0, s1, 0, s0);
            v4si v3 = _mm_load_si128((v4si*)(dst+0));
            // Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
            _mm_store_si128((v4si*)(dst+0), _mm_add_epi32(v3, _mm_madd_epi16(v0, v1)));
            dst+=4;
        }
        n = remain;
    }
#endif
    
    while(n--) {
        
        dst[0] += vol[0] * FETCH_SAMPLE(src, offset, increment);
        dst[1] += vol[1] * FETCH_SAMPLE(src, offset, increment);
        dst+=2;
    }
    return offset;
}


// Backported from drv_osx.c
void virtch_int8_to_fp(const int8_t* myInBuffer, float* myOutBuffer, int numChannels, size_t n)
{
    int i;
    const float f = (1.0f / 128.0f);
    
    if (numChannels == 1) {
        for (i = 0; i <(int)(n >> 1); i++) {
            myOutBuffer[1] = myOutBuffer[0] = (*myInBuffer++) * f;
            myOutBuffer+=2;
        }
    }
    else {
        for (i = 0; i < (int) n; i++) {
            *myOutBuffer++ = (*myInBuffer++) * f;
        }
    }
}


// Backported from drv_osx.c
void virtch_int16_to_fp(const sample_t* myInBuffer, float* myOutBuffer, int numChannels, size_t n)
{
    int i;
    const float f = (1.0f / 32768.0f);
    
#if (VMIX_SIMD == VMIX_SIMD_ALTIVEC)
    
    const vector float gain = vec_load_ps1(&f); /* multiplier */
    const vector float mix = vec_setzero();
    
    if (numChannels == 1) {
        int j = 0;
        /* TEST: OK */
        for (i = 0; i < n; i += 8, j += 16) {
            vector short int v0 = vec_ld(0, myInBuffer + i); /* Load 8 shorts */
            vector float v1 = vec_ctf((vector signed int)vec_unpackh(v0), 0); /* convert to float */
            vector float v2 = vec_ctf((vector signed int)vec_unpackl(v0), 0); /* convert to float */
            vector float v3 = vec_madd(v1, gain, mix); /* scale */
            vector float v4 = vec_madd(v2, gain, mix); /* scale */
            
            vector float v5 = vec_mergel(v3, v3); /* v3(0,0,1,1); */
            vector float v6 = vec_mergeh(v3, v3); /* v3(2,2,3,3); */
            vector float v7 = vec_mergel(v4, v4); /* v4(0,0,1,1); */
            vector float v8 = vec_mergeh(v4, v4); /* v4(2,2,3,3); */
            
            vec_st(v5, 0, myOutBuffer + j); /* Store 4 floats */
            vec_st(v6, 0, myOutBuffer + 4 + j); /* Store 4 floats */
            vec_st(v7, 0, myOutBuffer + 8 + j); /* Store 4 floats */
            vec_st(v8, 0, myOutBuffer + 12 + j); /* Store 4 floats */
        }
    }
    else {
        /* TEST: OK */
        for (i = 0; i < n; i += 8) {
            vector short int v0 = vec_ld(0, myInBuffer + i); /* Load 8 shorts */
            vector float v1 = vec_ctf((vector signed int)vec_unpackh(v0), 0); /* convert to float */
            vector float v2 = vec_ctf((vector signed int)vec_unpackl(v0), 0); /* convert to float */
            vector float v3 = vec_madd(v1, gain, mix); /* scale */
            vector float v4 = vec_madd(v2, gain, mix); /* scale */
            vec_st(v3, 0, myOutBuffer + i); /* Store 4 floats */
            vec_st(v4, 0, myOutBuffer + 4 + i); /* Store 4 floats */
        }
    }
    
#else
    
    if (numChannels == 1) {
        for (i = 0; i < (int)(n >> 1); i++) {
            myOutBuffer[1] = myOutBuffer[0] = (*myInBuffer++) * f;
            myOutBuffer+=2;
        }
    }
    else {
        for (i = 0; i < (int) n; i++) {
            *myOutBuffer++ = (*myInBuffer++) * f;
        }
    }
#endif
}


#ifdef TEST

/* Unit tests
 gcc virtch_simd.c -DTEST -o virtch -g -Os -mavx2
 mavx512f not working on XCode 6.2 : Apple didn't include the avx512fintrin.h
 
 */

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>

void *aligned_malloc( size_t size, int align )
{
    void *mem = malloc( size + (align-1) + sizeof(void*) );
    char *amem = ((char*)mem) + sizeof(void*);
    amem += align - ((uintptr_t)amem & (align - 1));
    ((void**)amem)[-1] = mem;
    return amem;
}

#define BEGIN startTime = mach_absolute_time();
#define END endTime = mach_absolute_time(); if (dt == 0 || endTime - startTime < dt) dt = endTime - startTime;


void aligned_free( void *mem )
{
    free( ((void**)mem)[-1] );
}

int main()
{
    uint64_t startTime;
    uint64_t endTime;
    
    int numSamples = 65536;
    int numChannels = 2;
    
    sample_t* dst = aligned_malloc(numSamples * 2 * numChannels, 64);
    int32_t* src = aligned_malloc(numSamples * 4 * numChannels, 64);
    
    fprintf(stdout, "====================\n");
    fprintf(stdout,"virtch_downmix_32_16 test... \n");
    
    int dt = 0;
    const int numLoop = 1000;
    for (int i = 0; i < numLoop; i++)
    {
        BEGIN
        virtch_downmix_32_16(src, dst, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles\n", dt);
    
    virtch_set_features(1);
    
    dt = 0;
    for (int i = 0; i < numLoop; i++)
    {
        BEGIN
        virtch_downmix_32_16(src, dst, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles (SIMD)\n", dt);
    
    
    fprintf(stdout, "====================\n");
    fprintf(stdout,"virtch_mix_stereo test... \n");
    
    int32_t vol[2] = {1<<BITSHIFT, 1<<BITSHIFT};
    virtch_set_features(0);
    dt = 0;
    {
        BEGIN
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles\n", dt);
    
    virtch_set_features(1);
    
    dt = 0;
    for (int i = 0; i < numLoop; i++)
    {
        BEGIN
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles (SIMD)\n", dt);
    
    
    fprintf(stdout, "====================\n");
    fprintf(stdout,"virtch_mix_stereo_st test... \n");
    
    virtch_set_features(0);
    dt = 0;
    {
        BEGIN
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles\n", dt);
    
    virtch_set_features(1);
    
    dt = 0;
    for (int i = 0; i < numLoop; i++)
    {
        BEGIN
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    
    
    fprintf(stdout, "%d cycles (SIMD)\n", dt);
    
    
}

#endif

