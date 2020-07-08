/*==============================================================================
 $Id$
 virtch_simd.c
 Stephane Denis
 V2.0
 Audio processing and resampling from 32bit (int and float) to 16bit sound buffer
 Implementation for SSE, Altivec, ARM Neon, AVX2, AVX512BW
 AVX2
 virtch_downmix_32_16          |52291    |6359     |+722 % faster
 virtch_mix_stereo             |179043   |79871    |+124 % faster
 virtch_mix_stereo_st          |85816    |33320    |+157 % faster
 virtch_pack_float_int16_st    |56495    |15414    |+266 % faster
 ==============================================================================*/
#include "virtch_simd.h"
#include  <memory.h>
#ifdef __ALTIVEC__
#ifdef __GNUC__
#include <ppc_intrinsics.h>
#endif
typedef vector float v4sf;  // vector of 4 float
typedef vector signed int v4si;  // vector of 4 uint32
#elif defined __ARM_NEON__
#ifdef _M_ARM64
#include <arm64_neon.h>
#else
#include <arm_neon.h>
#endif
typedef float32x4_t v4sf;  // vector of 4 float
typedef int32x4_t v4si;  // vector of 4 uint32
typedef int16x8_t v8sw;  // vector of 4 uint32
typedef int16x4_t v4sw;
#ifdef _WPHONE
// set 4 signed short to 16x4t
inline v4sw vec_set_4sw(int16_t a, int16_t b, int16_t c, int16_t d)
{
    v4sw ret;
    ret.n64_i16[0] = a;
    ret.n64_i16[1] = b;
    ret.n64_i16[2] = c;
    ret.n64_i16[3] = d;
    return ret;
}
#else
#define vec_set_4sw(a,b,c,d) {a, b, c, d}
#endif
#elif VMIX_SIMD == VMIX_SIMD_SSE ||  	  VMIX_SIMD == VMIX_SIMD_AVX || 	  VMIX_SIMD == VMIX_SIMD_AVX2 || 	  VMIX_SIMD == VMIX_SIMD_AVX512
// -mavx2
#ifdef __AVX__
FORCE_INLINE(v8si) vec256_fmadd_16(v8si a, v8si b, v8si c) // a  + b * c
{
    v8si res;
    // Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b.
    res = _mm256_mullo_epi16(b,  c); // not _mm_madd_epi16
    return _mm256_add_epi32 (res, a);
}
#endif
// -mavx512f
#ifdef __AVX512F__
FORCE_INLINE(v16si) vec512_fmadd_16(v16si a, v16si b, v16si c) // a  + b * c
{
    v16si res;
    // Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b.
    res = _mm512_mullo_epi16(b,  c); // not _mm_madd_epi16
    return _mm512_add_epi32 (res, a);
}
#endif
// Set 4 short integer to lo part
#define vec_setlo_4si(a, b, c, d)  _mm_set_epi16(0, d, 0, c, 0, b, 0, a)
// -msse4.1
FORCE_INLINE(v4si) vec_fmadd_16(v4si a, v4si b, v4si c) // a  + b * c
{
    v4si res;
    // Multiplies the 8 signed or unsigned 16-bit integers from a by the 8 signed or unsigned 16-bit integers from b.
    res = _mm_mullo_epi16(b,  c);
    return _mm_add_epi32 (res, a);
}
#endif
#ifdef VIRTCH_HQ
/**
*   Allow higher quality
*
* @param srce:
* @return none
*/
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
/**
 * enable SIMD function
 *
 * @param features: 1 or 0
 * @return none
 */
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
/**
 * downmix a 32bit buffer to a 8bit buffer
 *
 * @param dst: output buffer of samples
 * @param src: input buffer of samples
 * @param length: number of samples
 * @return none
 */
void virtch_downmix_32_8(const streamsample_t* src,
                         int8_t* dst,
                         size_t length)
{
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127);
            src++;
            length--;
            if (!length) break;
        }
    while(length>0) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127);
        src++;
        length--;
    }
}
#define FP_SHIFT 4
/**
 * downmix a 32bit buffer to a float-32bit buffer
 *
 * @param dst: output buffer of samples
 * @param src: input buffer of samples
 * @param length: number of samples
 * @return none
 */
void virtch_downmix_32_fp32(const streamsample_t* src,
                            float* dst,
                            size_t length)
{
    const float k = ((1.0f / 32768.0f) / (1 <<FP_SHIFT));
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            float sample = (*src++ >> (BITSHIFT-FP_SHIFT)) * k;
            *dst++ = CLAMP_F(sample);
            length--;
            if (!length) {
                return;
            }
        }
    while(length>0) {
        float sample = (*src++ >> (BITSHIFT-FP_SHIFT)) * k;
        *dst++ = CLAMP_F(sample);
        length--;
    }
}
/**
 * downmix a 32bit buffer to a 16bit buffer
 *
 * @param dst: output buffer of samples
 * @param src: input buffer of samples
 * @param length: number of samples
 * @return none
 */
void virtch_downmix_32_16(const streamsample_t* src,
                          int16_t* dst, // dst is REQUIRED to be aligned
                          size_t length) // length = number of sample (double this value for stereo)
{
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_16, 32767);
            src++;
            length--;
            if (!length) {
                return;
            }
        }
#if VMIX_SIMD == VMIX_SIMD_NEON
    int remain = length&7;
    if (virtch_features) {
        // Verified
        for (length>>=3; length; length--) {
            v4si v0 = vld1q_s32((int32_t*)(src+0));
            v4si v1 = vld1q_s32((int32_t*)(src+4));
            v4sw v2 = vqshrn_n_s32(v0, BITSHIFT_VOL_32_16);
            v4sw v3 = vqshrn_n_s32(v1, BITSHIFT_VOL_32_16);
            // Vector saturating narrow integer
            v4si v5 = vcombine_s16 (v2, v3);
            vst1q_s32((int32_t*)(dst+0), v5);
            dst+=8;
            src+=8;
        }
        length = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
    int remain = length&7;
    if (virtch_features) {
        // Untested, looks correct
        for (length>>=3; length; length--) {
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
        length = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_AVX512
    int remain = length&31;
    if (virtch_features) {
        // Untested, no hw access
        for (length>>=5; length; length--) {
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
        length = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_AVX2
    int remain = length&15;
    if (virtch_features) {
        // Untested, no hw access
        for (length>>=4; length; length--) {
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
        length = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_SSE || VMIX_SIMD == VMIX_SIMD_AVX
    int remain = length&7;
    if (virtch_features) {
        // Verified
        for (length>>=3; length; length--) {
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
        length = remain;
    }
#endif
    while(length>0) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_16, 32767);
        src++;
        length--;
    }
}
// mix dst, with 16bit sample.
size_t virtch_mix_mono(const sample_t* src,
                       const int32_t* vol,
                       streamsample_t* dst,
                       size_t offset,
                       size_t increment,
                       size_t length)
{
    sample_t sample;
    while(length--) {
        sample = FETCH_SAMPLE(src, offset, increment);
        *dst++ += vol[0] * sample;
    }
    return offset;
}
/**
 * mix a source buffer with volume, to a 32bit buffer. Apply pitch
 *
 * @param dst: output buffer of samples
 * @param src: input buffer of samples
 * @param vol: input volume (fixed point, 9 bit)
 * @param length: number of samples
 * @param offset: source offset
 * @param increment: increment (11 bit) for full speed
 * @return number of sample proceeded
 */
size_t virtch_mix_stereo(const sample_t* src,
                         const int32_t* vol,
                         streamsample_t* dst,
                         size_t offset,
                         size_t increment,
                         size_t length)
{
    sample_t sample;
    size_t remain = length;
    // dst can be misaligned ...
    if (virtch_features && length >= 4)
        while(!SAMPLE_ALIGNED(dst, 4)) {
            sample = FETCH_SAMPLE(src, offset, increment);
            *dst++ += vol[0] * sample;
            *dst++ += vol[1] * sample;
            length--;
            if (length == 0)
                break;
        }
    // src is always aligned ...
#if VMIX_SIMD >= VMIX_SIMD_AVX2
    if (virtch_features) {
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
        remain = length&7;
        for (length>>=3; length; length--) {
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
            _mm256_store_si256((v8si*)(dst+0), vec256_fmadd_16(v3, v0, v1));
            _mm256_store_si256((v8si*)(dst+8), vec256_fmadd_16(v4, v0, v2));
            dst+=16;
        }
    }
#elif VMIX_SIMD == VMIX_SIMD_NEON
    if (virtch_features) {
        // Verified
        // Unrolled 8 samples gave 10% speedup
        v4sw v0 = vec_set_4sw((int16_t)vol[0], (int16_t)vol[1], (int16_t)vol[0], (int16_t)vol[1] );
        remain = length&7;
        for (length>>=3; length; length--) {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            v4sw v1 = vec_set_4sw( s0, s0, s1, s1 );
            v4sw v2 = vec_set_4sw( s2, s2, s3, s3 );
            v4sw v5 = vec_set_4sw( s4, s4, s5, s5 );
            v4sw v6 = vec_set_4sw( s6, s6, s7, s7 );
            v4si v3 = vld1q_s32((int32_t*)(dst+0)); // src: a0, a1, a2, a3
            v4si v4 = vld1q_s32((int32_t*)(dst+4));
            v4si v7 = vld1q_s32((int32_t*)(dst+8)); // src: a0, a1, a2, a3
            v4si v8 = vld1q_s32((int32_t*)(dst+12));
            vst1q_s32((int32_t*)(dst+0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+4), vmlal_s16(v4, v0, v2)); // vmlal_lane_s16 ?
            vst1q_s32((int32_t*)(dst+8), vmlal_s16(v7, v0, v5)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+12), vmlal_s16(v8, v0, v6)); // vmlal_lane_s16 ?
            dst+=16;
        }
    }
#elif VMIX_SIMD == VMIX_SIMD_SSE || VMIX_SIMD == VMIX_SIMD_AVX
    if (virtch_features) {
        // 4
        v4si v0 = vec_setlo_4si((int16_t)vol[0], (int16_t)vol[1], (int16_t)vol[0], (int16_t)vol[1] );
        remain = length & 7;
        for (length>>=3; length; length--) {
            // Mono sample
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            v4si v1 = vec_setlo_4si( s0, s0, s1, s1 ); // 4 short
            v4si v2 = vec_setlo_4si( s2, s2, s3, s3 );
            v4si v5 = vec_setlo_4si( s4, s4, s5, s5 );
            v4si v6 = vec_setlo_4si( s6, s6, s7, s7 );
            v4si v3 = _mm_load_si128((v4si*) (dst +  0));
            v4si v4 = _mm_load_si128((v4si*) (dst +  4));
            v4si v7 = _mm_load_si128((v4si*) (dst +  8));
            v4si v8 = _mm_load_si128((v4si*) (dst + 12));
            // Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
            _mm_store_si128((v4si*)(dst+ 0), vec_fmadd_16(v3, v0, v1)); // r0 := (a0 * b0) + (a1 * b1)
            _mm_store_si128((v4si*)(dst+ 4), vec_fmadd_16(v4, v0, v2));
            _mm_store_si128((v4si*)(dst+ 8), vec_fmadd_16(v7, v0, v5));
            _mm_store_si128((v4si*)(dst+12), vec_fmadd_16(v8, v0, v6));
            dst+=16;
        }
    }
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
    if (virtch_features) {
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
        remain = length&3;
        for(length>>=2; length; length--) {
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
/**
 * mix a source buffer with volume, to a 32bit buffer. Apply pitch
 *
 * @param dst: output buffer of samples
 * @param src: input buffer of samples
 * @param vol: input volume (fixed point, 9 bit)
 * @param length: number of samples
 * @param offset: source offset
 * @param increment: increment (11 bit) for full speed
 * @return number of sample proceeded
 */
size_t virtch_mix_stereo_st(const sample_t* src,
                            const int32_t* vol,
                            streamsample_t* dst,
                            size_t offset,
                            size_t increment,
                            size_t length)
{
    if (virtch_features && length > 0) {
        while (!SAMPLE_ALIGNED(dst, 4)) {
            dst[0] += vol[0] * (streamsample_t)FETCH_SAMPLE(src, offset, increment);
            dst[1] += vol[1] * (streamsample_t)FETCH_SAMPLE(src, offset, increment);
            dst+=2;
            length--;
            if (!length)
                break;
        }
    }
#if VMIX_SIMD == VMIX_SIMD_NEON
    if (virtch_features) {
        // Verified
        v4sw v0 = vec_set_4sw((int16_t) vol[0], (int16_t) vol[1], (int16_t) vol[0], (int16_t) vol[1] );
        int remain = length&3;
        for (length>>=2; length; length--) {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            v4sw v1 = vec_set_4sw(s0, s1, s2, s3 );
            v4sw v4 = vec_set_4sw(s4, s5, s6, s7);
            v4si v3 = vld1q_s32((int32_t*)(dst + 0)); // src: a0, a1, a2, a3
            v4si v5 = vld1q_s32((int32_t*)(dst + 4)); // src: a0, a1, a2, a3
            vst1q_s32((int32_t*)(dst+0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
            vst1q_s32((int32_t*)(dst+4), vmlal_s16(v5, v0, v4)); // v3+= v0 * v1
            dst+=8;
        }
        length = remain;
    }
#elif VMIX_SIMD == VMIX_SIMD_SSE || VMIX_SIMD >= VMIX_SIMD_AVX
    // Stereo sample
    if (virtch_features) {
        // Verified
        v4si v0 =  vec_setlo_4si((int16_t) vol[0], (int16_t) vol[1], (int16_t) vol[0], (int16_t) vol[1] );
        int remain = length&3;
        for (length>>=3; length; length--) {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            v4si v1 = vec_setlo_4si(s0, s1, s2, s3);
            v4si v4 = vec_setlo_4si(s4, s5, s6, s7);
            v4si v3 = _mm_load_si128((v4si*) (dst + 0));
            v4si v5 = _mm_load_si128((v4si*) (dst + 4));
            _mm_store_si128((v4si*)(dst+0), vec_fmadd_16(v3, v0, v1)); // r0 := (a0 * b0) + (a1 * b1)
            _mm_store_si128((v4si*)(dst+4), vec_fmadd_16(v5, v0, v4)); // r0 := (a0 * b0) + (a1 * b1)
            dst+=8;
        }
        length = remain;
    }
#endif
    while(length--) {
        dst[0] += vol[0] * FETCH_SAMPLE(src, offset, increment);
        dst[1] += vol[1] * FETCH_SAMPLE(src, offset, increment);
        dst+=2;
    }
    return offset;
}
/**
 * Convert array of samples to float arrays
 *
 * @param dst: output buffer of samples
 * @param src:
 * @param channels: number of channels (1 or 2)
 * @param length: number of samples
 * @return none
 */
void virtch_int8_to_fp(const int8_t* src, float* dst, int numChannels, size_t length)
{
    int i;
    const float f = (1.0f / 128.0f);
    if (numChannels == 1) {
        for (i = 0; i <(int)(length >> 1); i++) {
            dst[1] = dst[0] = (*src++) * f;
            dst+=2;
        }
    } else {
        for (i = 0; i < (int) length; i++) {
            *dst++ = (*src++) * f;
        }
    }
}
/**
 * Convert array of samples to float arrays
 *
 * @param dst: output buffer of samples
 * @param src:
 * @param channels: number of channels (1 or 2)
 * @param length: number of samples
 * @return none
 */
void virtch_int16_to_fp(const sample_t* src, float* dst, int numChannels, size_t length)
{
    int i;
    const float f = (1.0f / 32768.0f);
#if (VMIX_SIMD == VMIX_SIMD_ALTIVEC)
    const vector float gain = vec_load_ps1(&f); /* multiplier */
    const vector float mix = vec_setzero();
    if (numChannels == 1) {
        int j = 0;
        /* TEST: OK */
        for (i = 0; i < length; i += 8, j += 16) {
            vector short int v0 = vec_ld(0, src + i); /* Load 8 shorts */
            vector float v1 = vec_ctf((vector signed int)vec_unpackh(v0), 0); /* convert to float */
            vector float v2 = vec_ctf((vector signed int)vec_unpackl(v0), 0); /* convert to float */
            vector float v3 = vec_madd(v1, gain, mix); /* scale */
            vector float v4 = vec_madd(v2, gain, mix); /* scale */
            vector float v5 = vec_mergel(v3, v3); /* v3(0,0,1,1); */
            vector float v6 = vec_mergeh(v3, v3); /* v3(2,2,3,3); */
            vector float v7 = vec_mergel(v4, v4); /* v4(0,0,1,1); */
            vector float v8 = vec_mergeh(v4, v4); /* v4(2,2,3,3); */
            vec_st(v5, 0, dst + j); /* Store 4 floats */
            vec_st(v6, 0, dst + 4 + j); /* Store 4 floats */
            vec_st(v7, 0, dst + 8 + j); /* Store 4 floats */
            vec_st(v8, 0, dst + 12 + j); /* Store 4 floats */
        }
    } else {
        /* TEST: OK */
        for (i = 0; i < length; i += 8) {
            vector short int v0 = vec_ld(0, src + i); /* Load 8 shorts */
            vector float v1 = vec_ctf((vector signed int)vec_unpackh(v0), 0); /* convert to float */
            vector float v2 = vec_ctf((vector signed int)vec_unpackl(v0), 0); /* convert to float */
            vector float v3 = vec_madd(v1, gain, mix); /* scale */
            vector float v4 = vec_madd(v2, gain, mix); /* scale */
            vec_st(v3, 0, dst + i); /* Store 4 floats */
            vec_st(v4, 0, dst + 4 + i); /* Store 4 floats */
        }
    }
#else
    if (numChannels == 1) {
        for (i = 0; i < (int)(length >> 1); i++) {
            dst[1] = dst[0] = (*src++) * f;
            dst+=2;
        }
    } else {
        for (i = 0; i < (int) length; i++) {
            *dst++ = (*src++) * f;
        }
    }
#endif
}
static const float ONE = 32767;
/**
 * Convert float to int with clamp
 *
 * @param inval: float value
 * @return integer value (16-bit signed)
 */
static sample_t virtch_ftoi(float inval)
{
    if (inval < -1.f)
        return -32768;
    if (inval > 1.f)
        return 32767;
    return (sample_t)(inval * ONE);
}
/**
 * packs two array of float (left and right) to an interleaved buffer. Useful for OGG Raw PCM Float Channel to 16-bit
 *
 * @param dst: output buffer of samples
 * @param left: input left channel of float
 * @param right: input right channel of float
 * @param length: number of samples
 * @return none
 */
static void virtch_pack_float_int16_st(sample_t* dst, const float* left, const float* right, size_t length)
{   
    size_t remain = length;
#if (VMIX_SIMD == VMIX_SIMD_AVX512)
    if (virtch_features)
        if ((((size_t)right & 63) == 0) && (((size_t)left & 63) == 0))
        {
            remain = length & 63;
            v16sf cst = _mm512_set1_ps(ONE); // This code was never tested
            for (length >>= 5; length; length--)
            {
                v16sf v0 = _mm512_mul_ps(_mm512_load_ps(left), cst);
                v16sf v1 = _mm512_mul_ps(_mm512_load_ps(right), cst);
                v16sf v2 = _mm512_mul_ps(_mm512_load_ps(left + 16), cst);
                v16sf v3 = _mm512_mul_ps(_mm512_load_ps(right + 16), cst);
                v16si v4 = _mm512_cvttps_epi32(v0);
                v16si v5 = _mm512_cvttps_epi32(v1);
                v16si v6 = _mm512_cvttps_epi32(v2);
                v16si v7 = _mm512_cvttps_epi32(v3);
                v16si v8 = _mm512_packs_epi32(v4, v6);
                v16si v9 = _mm512_packs_epi32(v5, v7);
                _mm512_store_si512((v16si*)(dst), _mm512_unpacklo_epi16(v8, v9));
                _mm512_store_si512((v16si*)(dst + 32), _mm512_unpackhi_epi16(v8, v9));
                left += 32;
                right += 32;
                dst += 64;
            }
            length = remain;
        }
#endif
#if (VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512)
    if (virtch_features)
        if ((((size_t)right & 31) == 0) && (((size_t)left & 31) == 0))
        {
            remain = length & 31;
            v8sf cst = _mm256_set1_ps(ONE);
            for (length >>= 4; length; length--)
            {
                v8sf v0 = _mm256_mul_ps(_mm256_load_ps(left), cst);
                v8sf v1 = _mm256_mul_ps(_mm256_load_ps(right), cst);
                v8sf v2 = _mm256_mul_ps(_mm256_load_ps(left + 8), cst);
                v8sf v3 = _mm256_mul_ps(_mm256_load_ps(right + 8), cst);
                v8si v4 = _mm256_cvttps_epi32(v0);
                v8si v6 = _mm256_cvttps_epi32(v2);
                v8si v5 = _mm256_cvttps_epi32(v1);
                v8si v7 = _mm256_cvttps_epi32(v3);
                v8si v8 = _mm256_packs_epi32(v4, v6);
                v8si v9 = _mm256_packs_epi32(v5, v7);
                unpacklo_epi16_store((v8si*)(dst), v8, v9);
                unpackhi_epi16_store((v8si*)(dst + 16), v8, v9);
                left += 16;
                right += 16;
                dst += 32;
            }
            length = remain;
            _mm256_zeroupper(); // Will switch to SSE
        }
#endif
#if (VMIX_SIMD >= VMIX_SIMD_SSE)
    if (virtch_features)
        if ((((size_t)right & 15) == 0) && (((size_t)left & 15) == 0))
        {
            v4sf cst = _mm_load_ps1(&ONE);
            remain = length & 31;
            for (length >>= 3; length; length--)
            {
                v4sf v0 = _mm_mul_ps(_mm_load_ps(left), cst);
                v4sf v1 = _mm_mul_ps(_mm_load_ps(right), cst);
                v4sf v2 = _mm_mul_ps(_mm_load_ps(left + 4), cst);
                v4sf v3 = _mm_mul_ps(_mm_load_ps(right + 4), cst);
                v4si v4 = _mm_cvttps_epi32(v0);
                v4si v5 = _mm_cvttps_epi32(v1);
                v4si v6 = _mm_cvttps_epi32(v2);
                v4si v7 = _mm_cvttps_epi32(v3);
                v4si v8 = _mm_packs_epi32(v4, v6);
                v4si v9 = _mm_packs_epi32(v5, v7);
                _mm_store_si128((v4si*)(dst), _mm_unpacklo_epi16(v8, v9));
                _mm_store_si128((v4si*)(dst + 8), _mm_unpackhi_epi16(v8, v9));
                left += 8;
                right += 8;
                dst += 16;
            }
            length = remain;
        }
#endif
    for (; length; dst += 2, left++, right++, length--)
    {
        dst[0] = virtch_ftoi(*left);
        dst[1] = virtch_ftoi(*right);
    }
    return;
}
/**
 * converts float array to sample_t
 *
 * @param dst: output buffer of samples
 * @param left: input left channel of float
 * @param length: number of samples
 * @return none
 */
static void virtch_pack_float_int16_mono(sample_t* dst, const float* left, size_t length)
{
#if (VMIX_SIMD == VMIX_SIMD_AVX512)
    if (((size_t)left & 63) == 0)
    {
        v16sf cst = _mm512_set1_ps(ONE);  // vbroadcastss
        size_t remain = length & 63;
        for (length >>= 5; length; length--)
        {
            v16sf v0 = _mm512_mul_ps(_mm512_load_ps(left), cst); // 16 samples
            v16sf v1 = _mm512_mul_ps(_mm512_load_ps(left + 16), cst); // 16  samples
            v16si v4 = _mm512_cvttps_epi32(v0);
            v16si v5 = _mm512_cvttps_epi32(v1);
            v16si v8 = _mm512_packs_epi32(v4, v5);
            _mm512_store_si512((v16si*)(dst), v8); // 32 samples
            left += 32;
            dst += 32;
        }
        length = remain;
    }
#endif
#if (VMIX_SIMD >= VMIX_SIMD_SSE)
    if (((size_t)left & 15) == 0)
    {
        v4sf cst = _mm_load_ps1(&ONE);
        size_t remain = length & 15;
        for (length >>= 3; length; length--)
        {
            v4sf v0 = _mm_mul_ps(_mm_load_ps(left), cst); // 4 samples
            v4sf v1 = _mm_mul_ps(_mm_load_ps(left + 4), cst); // 4 samples
            v4si v4 = _mm_cvttps_epi32(v0);
            v4si v5 = _mm_cvttps_epi32(v1);
            v4si v8 = _mm_packs_epi32(v4, v5);
            _mm_store_si128((v4si*)(dst), v8); // 8 samples
            left += 8;
            dst += 8;
        }
        length = remain;
    }
#endif
    for (; length; dst++, left++, length--)
    {
        dst[0] = virtch_ftoi(*left);
    }
    return;
}
/**
 * converts float array to sample_t
 *
 * @param dst: output buffer of samples
 * @param src: input channels
 * @param channels: number of channels (1 or 2)
 * @param length: number of samples
 * @return none
 */
int virtch_pack_float_int16(sample_t* dst, const float** src, int channels, size_t length)
{
#if (VMIX_SIMD >= VMIX_SIMD_SSE)
    _mm_prefetch((const char*)src[0], _MM_HINT_T0);
    if (channels == 2)
        _mm_prefetch((const char*)src[1], _MM_HINT_T0);
    if (channels == 2)
        virtch_pack_float_int16_st(dst, src[0], src[1], length);
    else if (channels == 1)
        virtch_pack_float_int16_mono(dst, src[0], length);
    return channels == 1 || channels == 2 ? (int)length : 0;
#else
    if (channels == 2)
    {
        const float* left = src[0], * right = src[1];
        for (; length; dst += 2, left++, right++, length--)
        {
            dst[0] = virtch_ftoi(*left);
            dst[1] = virtch_ftoi(*right);
        }
    }
    else if (channels == 1)
    {
        const float* mono = src[0];
        for (; length; dst++, mono++, length--)
        {
            dst[0] = virtch_ftoi(*mono);
        }
    }
    return channels == 1 || channels == 2 ? (int)length : 0;
#endif
}
/**
 * interleave two array of float to one
 *
 * @param dst: output buffer of samples
 * @param src: input channels
 * @param channels: number of channels (1 or 2)
 * @param length: number of samples
 * @return none
 */
static void virtch_deinterleave_st_float(float* dst, const float* left, const float* right, size_t length)
{
    size_t j;
    // 56964 cycles
    for (j = 0; j < length; j++)
    {
        dst[2 * j] = left[j];
        dst[2 * j + 1] = right[j];
    }
}
/**
 * interleave two array of float to one
 *
 * @param dst: output buffer of samples
 * @param src: input channels
 * @param channels: number of channels (1 or 2)
 * @param length: number of samples
 * @return none
 */
void virtch_deinterleave_float(float* dst, const float** src, int channels, size_t length)
{
#if (VMIX_SIMD == VMIX_SIMD_AVX || VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512 || VMIX_SIMD == VMIX_SIMD_SSE)
    _mm_prefetch((const char*)src[0], _MM_HINT_T0);
    if (channels == 2)
        _mm_prefetch((const char*)src[1], _MM_HINT_T0);
#endif
    if (channels == 1)
    {
        memcpy(dst, src[0], length * sizeof(float));
    }
    else if (channels == 2)
    {
        virtch_deinterleave_st_float(dst, src[0], src[1], length);
    }
}

#ifdef TEST
/***
 Unit tests
 gcc virtch_simd.c -DTEST -o virtch -g -Os -mavx2
 -mavx512bw or
 -mavx2 or
 -mavx or
 -sse4.2
 ***/
#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
void* aligned_malloc( size_t size, int align )
{
    void* mem = malloc( size + (align-1) + sizeof(void*) );
    char* amem = ((char*)mem) + sizeof(void*);
    amem += align - ((uintptr_t)amem & (align - 1));
    ((void**)amem)[-1] = mem;
    return amem;
}
#define BEGIN startTime = mach_absolute_time();
#define END endTime = mach_absolute_time(); dt = endTime - startTime;
void aligned_free( void* mem )
{
    free( ((void**)mem)[-1] );
}
#define dbgLog printf
int main()
{
    uint64_t startTime;
    uint64_t endTime;
    int numSamples = 65536;
    int numChannels = 2;
    dbgLog("# Test\t\t\t|Cycles (Ref.)\t\t|Cycles (SIMD)\t|Gain\n");
    sample_t* dst = aligned_malloc(numSamples * 2 * numChannels, 64);
    int32_t* src = aligned_malloc(numSamples * 4 * numChannels, 64);
    int dt = 0;
    int dtRef = 0;
    const int numLoop = 1000;
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_downmix_32_16(src, dst, numSamples);
        END
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_downmix_32_16(src, dst, numSamples);
        END
    }
    dbgLog("virtch_downmix_32_16\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt)-100);
    int32_t vol[2] = {1<<BITSHIFT, 1<<BITSHIFT};
    virtch_set_features(0);
    {
        BEGIN
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    dbgLog("virtch_mix_stereo\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt)-100);
    virtch_set_features(0);
    {
        BEGIN
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << FRACBITS, numSamples);
        END
    }
    dbgLog("virtch_mix_stereo_st\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt)-100);
    int length = 4096;
    float left[4096];
    float right[4096];
    virtch_set_features(0);
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_pack_float_int16_st(dst, left, right, length);
        END
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        BEGIN
        virtch_pack_float_int16_st(dst, left, right, length);
        END
    }
    dbgLog("virtch_pack_float_int16_st\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt)-100);
}
#endif
