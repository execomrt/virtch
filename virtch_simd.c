// virtch_simd.c
// V0.8 2024-11-11
//
// Copyright (c) 2024 RealTech VR
//
// This software is provided 'as-is', without any express or implied warranty.
// In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is provided to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "virtch_simd.h"
#include <memory.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

static const float ONE_16S = 32767;
static const float INV_ONE_16S = 1.0f / 32767.0f; 
static int virtch_features = 0;

/**
 * enable SIMD function
 *
 * @param features 1 or 0
 */
void virtch_set_features(int features)
{
   virtch_features = features;
}

#ifdef VIRTCH_HQ
/**
 * Allow higher quality
 *
 * @param srce:
 * @return none
 */
VMIX_FORCE_INLINE(sample_t) GetLerpSample(const sample_t* const srce, size_t idx)
{
    size_t i = idx>>FRACBITS;
    streamsample_t f = idx&FRACMASK;
    return (sample_t)(((((streamsample_t)srce[i+0] * (FRACMASK+1L-f)) +
                        ((streamsample_t)srce[i+1] * f)) >> FRACBITS));
}
#define FETCH_SAMPLE(src, offset, increment) GetLerpSample(src, offset); offset += increment
#else
#define FETCH_SAMPLE(src, offset, increment) ((sample_t)src[(offset) >> VMIX_FRACBITS]); offset += increment
#endif


/**
 * downmix a 32bit buffer to a 8bit buffer
 *
 * @param dst output buffer of samples
 * @param src  input buffer of samples
 * @param aLength number of samples
 */
void virtch_downmix_32_8(const streamsample_t* src,
                         int8_t* dst,
                         size_t aLength)
{
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127);
            src++;
            aLength--;
            if (!aLength) break;
        }
    while(aLength>0) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_8, 127);
        src++;
        aLength--;
    }
}

/**
 * downmix a 32bit buffer to a float-32bit buffer
 *
 * @param dst output buffer of samples
 * @param src input buffer of samples
 * @param aLength number of samples
 */
void virtch_downmix_32_fp32(const streamsample_t* src,
                            float* dst,
                            size_t aLength)
{
    const float k = ((1.0f / 32768.0f) / (1 <<VMIX_FP_SHIFT));
    if (virtch_features)
        while(!SAMPLE_ALIGNED(src, 4) || !SAMPLE_ALIGNED(dst, 4)) {
            float sample = (*src++ >> (VMIX_BITSHIFT-VMIX_FP_SHIFT)) * k;
            *dst++ = CLAMP_F(sample);
            aLength--;
            if (!aLength) {
                return;
            }
        }
    while(aLength>0) {
        float sample = (*src++ >> (VMIX_BITSHIFT-VMIX_FP_SHIFT)) * k;
        *dst++ = CLAMP_F(sample);
        aLength--;
    }
}
size_t virtch_downmix_32_16_neon(const streamsample_t* src,
                                      int16_t* dst,
                                      size_t aLength)
{
  int16_t* start = dst;

    // If SIMD features are enabled (NEON support)
#if VMIX_SIMD == VMIX_SIMD_NEON
    // Process data in 8-sample (128-bit) chunks using NEON
    for (aLength >>= 3; aLength; aLength--) {
        // Load 8 32-bit integers (4 bytes each) from src into NEON registers
        int32x4_t v0 = vld1q_s32((int32_t*)(src + 0));  // Load 4 samples
        int32x4_t v1 = vld1q_s32((int32_t*)(src + 4));  // Load next 4 samples

        // Saturating shift and narrow to 16-bit (vqshrn_n_s32 returns int16x4_t)
        int16x4_t v2 = vqshrn_n_s32(v0, BITSHIFT_VOL_32_16);  // First 4 samples
        int16x4_t v3 = vqshrn_n_s32(v1, BITSHIFT_VOL_32_16);  // Next 4 samples

        // Combine the two 16x4 registers into one 16x8 register
        int16x8_t v4 = vcombine_s16(v2, v3);  // Combine both halves into one register

        // Store the results in dst (8 samples, 128 bits at once)
        vst1q_s16(dst + 0, v4);

        // Move pointers forward by 8 samples (128-bit blocks)
        dst += 8;
        src += 8;
    }
#endif
	return dst - start;
    
}
static size_t virtch_downmix_32_16_sse(const streamsample_t* src,
    int16_t* dst,
    size_t aLength)
{
	int16_t* start = dst;
 
    // Process AVX512 (512 bits at a time)
#if VMIX_SIMD >= VMIX_SIMD_AVX512
    {
        size_t remaining = aLength - ((aLength >> 5) << 5); // Process in blocks of 32 samples
        for (aLength >>= 5; aLength; aLength--) {
            v16si v0 = _mm512_load_si512((v16si*)(src + 0));
            v16si v1 = _mm512_load_si512((v16si*)(src + 16));
            v16si v2 = _mm512_srai_epi32(v0, BITSHIFT_VOL_32_16);
            v16si v3 = _mm512_srai_epi32(v1, BITSHIFT_VOL_32_16);
            v16si v5 = _mm512_packs_epi32(v2, v3);
            _mm512_store_si512((v8si*)(dst + 0), v5);
            dst += 32;
            src += 32;
        }
        aLength = remaining; // Handle the remaining samples after the loop
    }
#endif

    // Process AVX2 (256 bits at a time)
#if VMIX_SIMD >= VMIX_SIMD_AVX2
    {
        size_t remaining = aLength - ((aLength >> 4) << 4); // Process in blocks of 16 samples
        for (aLength >>= 4; aLength; aLength--) {
            v8si v0 = _mm256_load_si256((v8si*)(src + 0));
            v8si v1 = _mm256_load_si256((v8si*)(src + 8));
            v8si v2 = _mm256_srai_epi32(v0, BITSHIFT_VOL_32_16);
            v8si v3 = _mm256_srai_epi32(v1, BITSHIFT_VOL_32_16);
            v8si v5 = _mm256_packs_epi32(v2, v3);
            _mm256_store_si256((v8si*)(dst + 0), v5);
            dst += 16;
            src += 16;
        }
		aLength = remaining; // Handle the remaining samples after the loop
    }
#endif

    // Process SSE (128 bits at a time)
#if VMIX_SIMD >= VMIX_SIMD_SSE
    for (aLength >>= 3; aLength; aLength--) {
        v4si v0 = _mm_load_si128((v4si*)(src + 0));
        v4si v1 = _mm_load_si128((v4si*)(src + 4));
        v4si v2 = _mm_srai_epi32(v0, BITSHIFT_VOL_32_16);
        v4si v3 = _mm_srai_epi32(v1, BITSHIFT_VOL_32_16);
        v4si v5 = _mm_packs_epi32(v2, v3);
        _mm_store_si128((v4si*)(dst + 0), v5);
        dst += 8;
        src += 8;
    }
#endif
    

	return dst - start;    
}

size_t virtch_downmix_32_16_ppc(const streamsample_t* src,
    int16_t* dst,
    size_t aLength)
{
	int16_t* start = dst;   
#if VMIX_SIMD == VMIX_SIMD_ALTIVEC
    for (aLength >>= 3; aLength; aLength--) {
        v4si v0 = vec_ld(0, (v4si*)(src + 0));
        v4si v1 = vec_ld(0, (v4si*)(src + 4));
        v4si v2 = vec_sra(v0, vec_splat_u32(BITSHIFT_VOL_32_16));
        v4si v3 = vec_sra(v1, vec_splat_u32(BITSHIFT_VOL_32_16));
        v4si v5 = vec_packs(v1, v2);
        vec_st(v5, 0, dst);
        dst += 8;
        src += 8;
    }
#endif
	return dst - start; 
}
/**
 * downmix a 32bit buffer to a 16bit buffer
 *
 * @param dst output buffer of samples
 * @param src input buffer of samples
 * @param aLength number of samples
 */
void virtch_downmix_32_16(const streamsample_t* src,
                          int16_t* dst, // dst is REQUIRED to be aligned
                          size_t aLength) // aLength = number of sample (double this value for stereo)
{
    int16_t* end = dst + aLength;
    
    if (virtch_features)
    {
#if VMIX_SIMD == VMIX_SIMD_NEON
        size_t p = virtch_downmix_32_16_neon(src, dst, aLength);
        src += p;
        dst += p;
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
        size_t p = virtch_downmix_32_16_ppc(src, dst, aLength);
        src += p;
        dst += p;
#elif VMIX_SIMD >= VMIX_SIMD_SSE
        size_t p = virtch_downmix_32_16_sse(src, dst, aLength);
        src += p;
        dst += p;
#endif
    }
    
    while(dst < end) {
        *dst++ = CLAMP((*src) >> BITSHIFT_VOL_32_16, 32767);
        src++;
    }
    

}
// mix dst, with 16bit sample.
size_t virtch_mix_mono(const sample_t* src,
                       const int32_t* vol,
                       streamsample_t* dst,
                       size_t offset,
                       size_t increment,
                       size_t aLength)
{
    sample_t sample;
    while(aLength--) {
        sample = FETCH_SAMPLE(src, offset, increment);
        *dst++ += vol[0] * sample;
    }
    return offset;
}

static size_t virtch_mix_stereo_sse(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
    streamsample_t* end = dst + aLength * 2;
    streamsample_t* start = dst;
#if VMIX_SIMD >= VMIX_SIMD_SSE
        
    // Define SSE volume multiplier
    __m128i vol_0 = _mm_set1_epi32(vol[0]);
    __m128i vol_1 = _mm_set1_epi32(vol[1]);

#if VMIX_SIMD >= VMIX_SIMD_AVX2
    {
        __m256i vol_0 = _mm256_set1_epi32(vol[0]);
        __m256i vol_1 = _mm256_set1_epi32(vol[1]);

        size_t remaining = aLength - ((aLength >> 5) << 5); // Process in blocks of 32 samples
        for (aLength >>= 5; aLength; aLength--)
        {
            // Fetch samples from src in sets of 8 using the FETCH_SAMPLE macro
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            sample_t s8 = FETCH_SAMPLE(src, offset, increment);
            sample_t s9 = FETCH_SAMPLE(src, offset, increment);
            sample_t s10 = FETCH_SAMPLE(src, offset, increment);
            sample_t s11 = FETCH_SAMPLE(src, offset, increment);
            sample_t s12 = FETCH_SAMPLE(src, offset, increment);
            sample_t s13 = FETCH_SAMPLE(src, offset, increment);
            sample_t s14 = FETCH_SAMPLE(src, offset, increment);
            sample_t s15 = FETCH_SAMPLE(src, offset, increment);

            // Load the samples into AVX2 registers in pairs
            __m256i s0_7 = _mm256_set_epi32(s7, s6, s5, s4, s3, s2, s1, s0);
            __m256i s8_15 = _mm256_set_epi32(s15, s14, s13, s12, s11, s10, s9, s8);

            // Multiply each set of samples by the respective volume
            __m256i result_0 = _mm256_mullo_epi32(s0_7, vol_0); // Left channel for s0 to s7
            __m256i result_1 = _mm256_mullo_epi32(s0_7, vol_1); // Right channel for s0 to s7
            __m256i result_2 = _mm256_mullo_epi32(s8_15, vol_0); // Left channel for s8 to s15
            __m256i result_3 = _mm256_mullo_epi32(s8_15, vol_1); // Right channel for s8 to s15

            // Store results back into destination array
            _mm256_storeu_si256((__m256i*)(dst), result_0);       // Store s0 to s7 (left)
            _mm256_storeu_si256((__m256i*)(dst + 8), result_1);   // Store s0 to s7 (right)
            _mm256_storeu_si256((__m256i*)(dst + 16), result_2);  // Store s8 to s15 (left)
            _mm256_storeu_si256((__m256i*)(dst + 24), result_3);  // Store s8 to s15 (right)

            dst += 32;
        }
        aLength = remaining; // Handle the remaining samples after the loop
    }

#endif
    {
        size_t remaining = aLength - ((aLength >> 4) << 4);
        for (aLength >>= 4; aLength; aLength--)
        {
            // Fetch samples from src using the FETCH_SAMPLE macro
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);

            // Load the samples into SSE registers
            __m128i s0_1 = _mm_set_epi32(0, 0, s1, s0); // s0, s1 (pair)
            __m128i s2_3 = _mm_set_epi32(0, 0, s3, s2); // s2, s3 (pair)
            __m128i s4_5 = _mm_set_epi32(0, 0, s5, s4); // s4, s5 (pair)
            __m128i s6_7 = _mm_set_epi32(0, 0, s7, s6); // s6, s7 (pair)

            // Multiply each sample by volume for left and right channels
            __m128i result_0 = _mm_mullo_epi32(s0_1, vol_0); // Left channel for s0, s1
            __m128i result_1 = _mm_mullo_epi32(s2_3, vol_1); // Right channel for s2, s3
            __m128i result_2 = _mm_mullo_epi32(s4_5, vol_0); // Left channel for s4, s5
            __m128i result_3 = _mm_mullo_epi32(s6_7, vol_1); // Right channel for s6, s7

            // Store results back into destination array
            _mm_storeu_si128((__m128i*)(dst), result_0);  // Store s0, s1
            _mm_storeu_si128((__m128i*)(dst + 4), result_1);  // Store s2, s3
            _mm_storeu_si128((__m128i*)(dst + 8), result_2);  // Store s4, s5
            _mm_storeu_si128((__m128i*)(dst + 12), result_3);  // Store s6, s7

            dst += 16;
        }
        aLength = remaining;
    }
#endif

    while (dst < end) {
        streamsample_t sample = FETCH_SAMPLE(src, offset, increment);
        dst[0] += vol[0] * sample;
        dst[1] += vol[1] * sample;
        dst += 2;
    }
    return dst - start;
}

size_t virtch_mix_stereo_neon(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
    streamsample_t* start = dst;
    streamsample_t* end = dst + aLength * 2;
#if VMIX_SIMD == VMIX_SIMD_NEON
    v4sw v0 = vec_set_4sw((int16_t)vol[0], (int16_t)vol[1], (int16_t)vol[0], (int16_t)vol[1]);
    for (aLength >>= 4; aLength; aLength--) {
        sample_t s0 = FETCH_SAMPLE(src, offset, increment);
        sample_t s1 = FETCH_SAMPLE(src, offset, increment);
        sample_t s2 = FETCH_SAMPLE(src, offset, increment);
        sample_t s3 = FETCH_SAMPLE(src, offset, increment);
        sample_t s4 = FETCH_SAMPLE(src, offset, increment);
        sample_t s5 = FETCH_SAMPLE(src, offset, increment);
        sample_t s6 = FETCH_SAMPLE(src, offset, increment);
        sample_t s7 = FETCH_SAMPLE(src, offset, increment);
        v4sw v1 = vec_set_4sw(s0, s0, s1, s1);
        v4sw v2 = vec_set_4sw(s2, s2, s3, s3);
        v4sw v5 = vec_set_4sw(s4, s4, s5, s5);
        v4sw v6 = vec_set_4sw(s6, s6, s7, s7);
        v4si v3 = vld1q_s32((int32_t*)(dst + 0)); // src: a0, a1, a2, a3
        v4si v4 = vld1q_s32((int32_t*)(dst + 4));
        v4si v7 = vld1q_s32((int32_t*)(dst + 8)); // src: a0, a1, a2, a3
        v4si v8 = vld1q_s32((int32_t*)(dst + 12));
        vst1q_s32((int32_t*)(dst + 0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
        vst1q_s32((int32_t*)(dst + 4), vmlal_s16(v4, v0, v2)); // vmlal_lane_s16 ?
        vst1q_s32((int32_t*)(dst + 8), vmlal_s16(v7, v0, v5)); // v3+= v0 * v1
        vst1q_s32((int32_t*)(dst + 12), vmlal_s16(v8, v0, v6)); // vmlal_lane_s16 ?
        dst += 16;
    }
#endif

    while (dst < end) {
        streamsample_t sample = FETCH_SAMPLE(src, offset, increment);
        dst[0] += vol[0] * sample;
        dst[1] += vol[1] * sample;
        dst += 2;
    }

    return dst - start;
}

size_t virtch_mix_stereo_ppc(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
    streamsample_t* start = dst;
    streamsample_t* end = dst + aLength * 2;
#if VMIX_SIMD == VMIX_SIMD_ALTIVEC
    // Untested, from virtch.c
    int32_t volq[] = { vol[0], vol[1], vol[0], vol[1] };
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
    for (aLength >>= 2; aLength; aLength--) {
        // Load constants
        s[0] = FETCH_SAMPLE(src, offset, increment);
        s[1] = FETCH_SAMPLE(src, offset, increment);
        s[2] = FETCH_SAMPLE(src, offset, increment);
        s[3] = FETCH_SAMPLE(src, offset, increment);
        s[4] = 0;
        vector short int r1 = vec_ld(0, s);
        vector signed short v1 = vec_perm(r1, r1, (vector unsigned char)(0 * 2, 0 * 2 + 1, // s0
            4 * 2, 4 * 2 + 1, // 0
            0 * 2, 0 * 2 + 1, // s0
            4 * 2, 4 * 2 + 1, // 0
            1 * 2, 1 * 2 + 1, // s1
            4 * 2, 4 * 2 + 1, // 0
            1 * 2, 1 * 2 + 1, // s1
            4 * 2, 4 * 2 + 1  // 0
            ));
        vector signed short v2 = vec_perm(r1, r1, (vector unsigned char)(2 * 2, 2 * 2 + 1, // s2
            4 * 2, 4 * 2 + 1, // 0
            2 * 2, 2 * 2 + 1, // s2
            4 * 2, 4 * 2 + 1, // 0
            3 * 2, 3 * 2 + 1, // s3
            4 * 2, 4 * 2 + 1, // 0
            3 * 2, 3 * 2 + 1, // s3
            4 * 2, 4 * 2 + 1  // 0
            ));
        vector signed int v3 = vec_ld(0, dst);
        vector signed int v4 = vec_ld(0, dst + 4);
        vector signed int v5 = vec_mule(v0, v1);
        vector signed int v6 = vec_mule(v0, v2);
        vec_st(vec_add(v3, v5), 0, dst);
        vec_st(vec_add(v4, v6), 0x10, dst);
        dst += 8;
}
#endif

    while (dst < end) {
        streamsample_t sample = FETCH_SAMPLE(src, offset, increment);
        dst[0] += vol[0] * sample;
        dst[1] += vol[1] * sample;
        dst += 2;
    }
    return dst - start;
}

/**
 * mix a source buffer with volume, to a 32bit buffer. Apply pitch
 *
 * @param dst output buffer of samples
 * @param src input buffer of samples
 * @param vol input volume (fixed point, 9 bit)
 * @param aLength number of samples (divide by 2 from bytes count)
 * @param offset source offset
 * @param increment increment (11 bit) for full speed
 * @return number of sample proceeded
 */
size_t virtch_mix_stereo(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
    streamsample_t* end = dst + (aLength << 1);
    if (virtch_features)
    {
#if VMIX_SIMD >= VMIX_SIMD_SSE    
        virtch_mix_stereo_sse(src, vol, dst, offset, increment, aLength);            
#elif VMIX_SIMD == VMIX_SIMD_NEON
        virtch_mix_stereo_neon(src, vol, dst, offset, increment, aLength);
#elif VMIX_SIMD == VMIX_SIMD_ALTIVEC
        virtch_mix_stereo_ppc(src, vol, dst, offset, increment, aLength);
#endif
    }
    else
    {        
        sample_t sample;
        while (dst < end) {
            sample = FETCH_SAMPLE(src, offset, increment);
            dst[0] += vol[0] * sample;
            dst[1] += vol[1] * sample;
            dst += 2;
        }
    }
    return offset;
}

static size_t virtch_mix_stereo_st_sse(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
	streamsample_t* start = dst;
    streamsample_t* end = dst + (aLength << 1);

#if VMIX_SIMD >= VMIX_SIMD_AVX2
    if (aLength > 0)
    {
        // Define AVX2 volume multipliers
        __m256i vol_0 = _mm256_set1_epi32(vol[0]);
        __m256i vol_1 = _mm256_set1_epi32(vol[1]);

        size_t remaining = aLength - ((aLength >> 4) << 4);  // Process in blocks of 16 samples
        for (aLength >>= 4; aLength; aLength--)
        {
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);
            sample_t s8 = FETCH_SAMPLE(src, offset, increment);
            sample_t s9 = FETCH_SAMPLE(src, offset, increment);
            sample_t s10 = FETCH_SAMPLE(src, offset, increment);
            sample_t s11 = FETCH_SAMPLE(src, offset, increment);
            sample_t s12 = FETCH_SAMPLE(src, offset, increment);
            sample_t s13 = FETCH_SAMPLE(src, offset, increment);
            sample_t s14 = FETCH_SAMPLE(src, offset, increment);
            sample_t s15 = FETCH_SAMPLE(src, offset, increment);

            // Load samples into AVX2 registers
            __m256i s0_7 = _mm256_set_epi32(s7, s6, s5, s4, s3, s2, s1, s0);  // First 8 samples
            __m256i s8_15 = _mm256_set_epi32(s15, s14, s13, s12, s11, s10, s9, s8);  // Next 8 samples

            // Load destination values to add to
            __m256i dst0 = _mm256_loadu_si256((__m256i*)(dst + 0));  // Load first 8 destination samples
            __m256i dst1 = _mm256_loadu_si256((__m256i*)(dst + 8));  // Load next 8 destination samples

            // Multiply samples by the volume and add to destination
            __m256i result_0 = _mm256_add_epi32(dst0, _mm256_mullo_epi32(s0_7, vol_0));  // Apply left channel volume to s0 to s7
            __m256i result_1 = _mm256_add_epi32(dst1, _mm256_mullo_epi32(s8_15, vol_1));  // Apply right channel volume to s8 to s15

            // Store results back to destination
            _mm256_storeu_si256((__m256i*)(dst + 0), result_0);  // Store results for s0 to s7
            _mm256_storeu_si256((__m256i*)(dst + 8), result_1);  // Store results for s8 to s15

            dst += 16;  // Move to the next set of samples
        }
        aLength = remaining;  // Handle any remaining samples after the loop
    }
#endif
#if VMIX_SIMD >= VMIX_SIMD_SSE
    if (aLength > 0)
    {
        __m128i vol_0 = _mm_set1_epi32(vol[0]);
        __m128i vol_1 = _mm_set1_epi32(vol[1]);

        for (aLength >>= 3; aLength; aLength--) {
            // Fetch 8 samples from src using FETCH_SAMPLE macro
            sample_t s0 = FETCH_SAMPLE(src, offset, increment);
            sample_t s1 = FETCH_SAMPLE(src, offset, increment);
            sample_t s2 = FETCH_SAMPLE(src, offset, increment);
            sample_t s3 = FETCH_SAMPLE(src, offset, increment);
            sample_t s4 = FETCH_SAMPLE(src, offset, increment);
            sample_t s5 = FETCH_SAMPLE(src, offset, increment);
            sample_t s6 = FETCH_SAMPLE(src, offset, increment);
            sample_t s7 = FETCH_SAMPLE(src, offset, increment);

            // Load samples into SSE registers
            __m128i s0_3 = _mm_set_epi32(s3, s2, s1, s0); // Set s0, s1, s2, s3
            __m128i s4_7 = _mm_set_epi32(s7, s6, s5, s4); // Set s4, s5, s6, s7

            // Load destination values to add to
            __m128i dst0 = _mm_loadu_si128((__m128i*)(dst + 0)); // Load first 4 samples
            __m128i dst1 = _mm_loadu_si128((__m128i*)(dst + 4)); // Load next 4 samples

            // Multiply samples with the volume for each channel
            __m128i result_0 = _mm_add_epi32(dst0, _mm_mullo_epi32(s0_3, vol_0)); // Apply left channel volume
            __m128i result_1 = _mm_add_epi32(dst1, _mm_mullo_epi32(s4_7, vol_1)); // Apply right channel volume

            // Store results back to destination
            _mm_storeu_si128((__m128i*)(dst + 0), result_0); // Store result for s0 to s3
            _mm_storeu_si128((__m128i*)(dst + 4), result_1); // Store result for s4 to s7

            dst += 8;  // Move to next set of samples
        }
    }
#endif
    while (dst < end) {
        dst[0] += vol[0] * FETCH_SAMPLE(src, offset, increment);
        dst[1] += vol[1] * FETCH_SAMPLE(src, offset, increment);
        dst += 2;
    }
	return dst - start; 
}


size_t virtch_mix_stereo_st_neon(const sample_t* src,
    const int32_t* vol,
    streamsample_t* dst,
    size_t offset,
    size_t increment,
    size_t aLength)
{
    streamsample_t* start = dst;   
    streamsample_t* end = dst + (aLength << 1);

#if VMIX_SIMD == VMIX_SIMD_NEON
    v4sw v0 = vec_set_4sw((int16_t)vol[0], (int16_t)vol[1], (int16_t)vol[0], (int16_t)vol[1]);
    for (aLength >>= 3; aLength; aLength--) {
        sample_t s0 = FETCH_SAMPLE(src, offset, increment);
        sample_t s1 = FETCH_SAMPLE(src, offset, increment);
        sample_t s2 = FETCH_SAMPLE(src, offset, increment);
        sample_t s3 = FETCH_SAMPLE(src, offset, increment);
        sample_t s4 = FETCH_SAMPLE(src, offset, increment);
        sample_t s5 = FETCH_SAMPLE(src, offset, increment);
        sample_t s6 = FETCH_SAMPLE(src, offset, increment);
        sample_t s7 = FETCH_SAMPLE(src, offset, increment);
        v4sw v1 = vec_set_4sw(s0, s1, s2, s3);
        v4sw v4 = vec_set_4sw(s4, s5, s6, s7);
        v4si v3 = vld1q_s32((int32_t*)(dst + 0)); // src: a0, a1, a2, a3
        v4si v5 = vld1q_s32((int32_t*)(dst + 4)); // src: a0, a1, a2, a3
        vst1q_s32((int32_t*)(dst + 0), vmlal_s16(v3, v0, v1)); // v3+= v0 * v1
        vst1q_s32((int32_t*)(dst + 4), vmlal_s16(v5, v0, v4)); // v3+= v0 * v1
        dst += 8;
    }
#endif
    while (dst < end) {
        dst[0] += vol[0] * FETCH_SAMPLE(src, offset, increment);
        dst[1] += vol[1] * FETCH_SAMPLE(src, offset, increment);
        dst += 2;
    }
    return dst - start;
}

size_t virtch_mix_stereo_st(const sample_t* __src,
                            const int32_t* vol,
                            streamsample_t* __dst,
                            size_t offset,
                            size_t increment,
                            size_t aLength)
{
    streamsample_t* dst = (streamsample_t*)ALIGNED_AS(__dst, 64);
    const sample_t* src = (const sample_t*)ALIGNED_AS(__src, 16);
    streamsample_t* end = dst + (aLength << 1);


    if (virtch_features) {
#if VMIX_SIMD >= VMIX_SIMD_SSE
        virtch_mix_stereo_st_sse(src, vol, dst, offset, increment, aLength);
#elif VMIX_SIMD == VMIX_SIMD_NEON
        virtch_mix_stereo_st_neon(src, vol, dst, offset, increment, aLength);
#endif
    }
    else
    {
        while (dst < end) {
            dst[0] += vol[0] * FETCH_SAMPLE(src, offset, increment);
            dst[1] += vol[1] * FETCH_SAMPLE(src, offset, increment);
            dst += 2;
        }
    }
  
    return offset;
}
/**
 * Convert array of samples to float arrays
 *
 */
void virtch_int8_to_fp(const int8_t* src, float* dst, int numChannels, size_t aLength)
{
    int i;
    const float f = (1.0f / 128.0f);
    if (numChannels == 1) {
        for (i = 0; i <(int)(aLength >> 1); i++) {
            dst[1] = dst[0] = (*src++) * f;
            dst+=2;
        }
    } else {
        for (i = 0; i < (int) aLength; i++) {
            *dst++ = (*src++) * f;
        }
    }
}

void virtch_int16_to_ppc(const sample_t* src, float* dst, int numChannels, size_t aLength)
{
#if (VMIX_SIMD == VMIX_SIMD_ALTIVEC)

    const vector float gain = vec_load_ps1(&INV_ONE_16S); /* multiplier */
    const vector float mix = vec_setzero();
    if (numChannels == 1) {
        int j = 0;
        /* TEST: OK */
        for (i = 0; i < aLength; i += 8, j += 16) {
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
    }
    else {
        /* TEST: OK */
        for (i = 0; i < aLength; i += 8) {
            vector short int v0 = vec_ld(0, src + i); /* Load 8 shorts */
            vector float v1 = vec_ctf((vector signed int)vec_unpackh(v0), 0); /* convert to float */
            vector float v2 = vec_ctf((vector signed int)vec_unpackl(v0), 0); /* convert to float */
            vector float v3 = vec_madd(v1, gain, mix); /* scale */
            vector float v4 = vec_madd(v2, gain, mix); /* scale */
            vec_st(v3, 0, dst + i); /* Store 4 floats */
            vec_st(v4, 0, dst + 4 + i); /* Store 4 floats */
        }
    }
#endif
}
/**
 * Convert array of samples to float arrays
 *
 */
void virtch_int16_to_fp(const sample_t* src, float* dst, int numChannels, size_t aLength)
{
    int i;
    
#if (VMIX_SIMD == VMIX_SIMD_ALTIVEC)
    if (virtch_features)
		virtch_int16_to_ppc(src, dst, numChannels, aLength);    
#else
    if (numChannels == 1) {
        for (i = 0; i < (int)(aLength >> 1); i++) {
            dst[1] = dst[0] = (*src++) * INV_ONE_16S;
            dst+=2;
        }
    } else {
        for (i = 0; i < (int) aLength; i++) {
            *dst++ = (*src++) * INV_ONE_16S;
        }
    }
#endif
}

static sample_t virtch_ftoi(float inval)
{
    if (inval < -1.f)
        return -32768;
    if (inval > 1.f)
        return 32767;
    return (sample_t)(inval * ONE_16S);
}

inline size_t process_simd_floats_to_ints_st(const float* left, const float* right, sample_t* dst, size_t aLength) {

    sample_t* start = dst;
#if (VMIX_SIMD >= VMIX_SIMD_AVX512)        
    {
        v16sf cst = _mm512_set1_ps(ONE_16S); // This code was never tested
        size_t remainder = aLength - ((aLength >> 5) << 5);
        for (aLength >>= 5; aLength; aLength--)
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
        aLength = remainder;
     }
#endif
#if (VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512)
    {
        v8sf cst = _mm256_set1_ps(ONE_16S);
        size_t remainder = aLength - ((aLength >> 4) << 4);
        for (aLength >>= 4; aLength; aLength--)
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
        aLength = remainder;
        _mm256_zeroupper(); // Will switch to SSE
    }
#endif
#if (VMIX_SIMD >= VMIX_SIMD_SSE)
    {
        v4sf cst = _mm_load_ps1(&ONE_16S);
        size_t remainder = aLength - ((aLength >> 3) << 3);
        for (aLength >>= 3; aLength; aLength--)
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
        aLength = remainder;
    }
#endif
#if (VMIX_SIMD == VMIX_SIMD_NEON)
    {
        float32x4_t cst = vdupq_n_f32(ONE_16S);  // Load constant ONE_16S into NEON register
        for (aLength >>= 3; aLength; aLength--)
        {
            // Load 4 floats from left and right arrays
            float32x4_t v0 = vmulq_f32(vld1q_f32(left), cst);
            float32x4_t v1 = vmulq_f32(vld1q_f32(right), cst);
            float32x4_t v2 = vmulq_f32(vld1q_f32(left + 4), cst);
            float32x4_t v3 = vmulq_f32(vld1q_f32(right + 4), cst);
            int32x4_t v4 = vcvtq_s32_f32(v0);
            int32x4_t v5 = vcvtq_s32_f32(v1);
            int32x4_t v6 = vcvtq_s32_f32(v2);
            int32x4_t v7 = vcvtq_s32_f32(v3);
            vst1q_s16((int16_t*)(dst), vqmovn_high_s32(vqmovn_s32(v4), v6));
            vst1q_s16((int16_t*)(dst + 8), vqmovn_high_s32(vqmovn_s32(v5), v7));
            left += 8;
            right += 8;
            dst += 16;
        }
    }
#endif

    return dst - start;
}



/**
 * converts float array to sample
 *
 * @param dst output buffer of samples
 * @param left input left channel of float
 * @param aLength number of samples
 */

inline size_t process_simd_floats_to_ints(const float* left, sample_t* dst, size_t aLength)
{
    sample_t* start = dst;
#if (VMIX_SIMD == VMIX_SIMD_AVX512)
    {
        v16sf cst = _mm512_set1_ps(ONE_16S);  // vbroadcastss
        for (aLength >>= 5; aLength; aLength--)
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
    }
#elif (VMIX_SIMD >= VMIX_SIMD_SSE)    
    {
        v4sf cst = _mm_load_ps1(&ONE_16S);
        for (aLength >>= 3; aLength; aLength--)
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

    }
#endif

    return dst - start;
}

size_t virtch_pack_float_int16_st(sample_t* __dst, const float* __left, const float* __right, size_t aLength) {
    sample_t* dst = (sample_t*)ALIGNED_AS(__dst, 64);  // Ensure destination is aligned to 64 bytes
    float* left = (float*)ALIGNED_AS(__left, 16);       // Ensure left array is aligned to 16 bytes
    float* right = (float*)ALIGNED_AS(__right, 16);     // Ensure right array is aligned to 16 bytes
    sample_t* end = dst + aLength * 2;

    // Check virtch_features first to enable SIMD
    if (virtch_features) {
        
		size_t processed = process_simd_floats_to_ints_st(left, right, dst, aLength); // Process SIMD operations (if available 
		if (processed == aLength * 2) return aLength; // If all samples are processed, return (no need to process non-SIMD path
        dst += processed;
		left += processed >> 1;
        right += processed >> 1;
    }

    // Fallback for non-SIMD paths when virtch_features is disabled or SIMD isn't used
    while (dst < end) {
        dst[0] = virtch_ftoi(*left);  // Convert float to int16_t for left channel
        dst[1] = virtch_ftoi(*right); // Convert float to int16_t for right channel
        dst += 2;
        left++;
        right++;
    }
    return aLength;
}

/**
 * converts float array to sample
 *
 * @param dst output buffer of samples
 * @param left input left channel of float
 * @param aLength number of samples
 */

static void virtch_pack_float_int16_mono(sample_t* dst,
    const float* left,
    size_t aLength)
{
    sample_t* end = dst + aLength;

    // Check virtch_features first to enable SIMD
    if (virtch_features) {

        size_t processed = process_simd_floats_to_ints(left, dst, aLength); // Process SIMD operations (if available 
        if (processed == aLength ) return; // If all samples are processed, return (no need to process non-SIMD path
        dst += processed;
        left += processed >> 1;
    }
    
    while(dst < end)
    {
        dst[0] = virtch_ftoi(*left);
        dst++;
        left++;
    }
    return;
}

/**
 * converts float array to sample
 *
 * @param dst output buffer of samples
 * @param src input channels
 * @param channels number of channels (1 or 2)
 * @param aLength number of samples
 */
size_t virtch_pack_float_int16(sample_t* dst,
    const float** src,
    int channels,
    size_t aLength)
{
#if (VMIX_SIMD >= VMIX_SIMD_SSE)
    _mm_prefetch((const char*)src[0], _MM_HINT_T0);
    if (channels == 2)
        _mm_prefetch((const char*)src[1], _MM_HINT_T0);
    if (channels == 2)
        virtch_pack_float_int16_st(dst, src[0], src[1], aLength);
    else if (channels == 1)
        virtch_pack_float_int16_mono(dst, src[0], aLength);
    return channels == 1 || channels == 2 ? aLength : 0;
#elif (VMIX_SIMD == VMIX_SIMD_NEON)
    if (channels == 2)
        virtch_pack_float_int16_st(dst, src[0], src[1], aLength);
    else if (channels == 1)
        virtch_pack_float_int16_mono(dst, src[0], aLength);
    return channels == 1 || channels == 2 ? aLength : 0;
#else
    if (channels == 2)
    {
        const float* left = src[0], * right = src[1];
        for (; aLength; dst += 2, left++, right++, aLength--)
        {
            dst[0] = virtch_ftoi(*left);
            dst[1] = virtch_ftoi(*right);
        }
    }
    else if (channels == 1)
    {
        const float* mono = src[0];
        for (; aLength; dst++, mono++, aLength--)
        {
            dst[0] = virtch_ftoi(*mono);
        }
    }
    return channels == 1 || channels == 2 ? (int)aLength : 0;
#endif
}
/**
 * interleave two array of float to one
 *
 * @param dst output buffer of samples
 * @param left input left channel of float
 * @param right input right channel of float
 * @param aLength number of samples
 */
static void virtch_deinterleave_st_float(float* dst, const float* left, const float* right, size_t aLength)
{
    size_t j;
    // 56964 cycles
    for (j = 0; j < aLength; j++)
    {
        dst[2 * j] = left[j];
        dst[2 * j + 1] = right[j];
    }
}
/**
 * interleave two array of float to one
 *
 * @param dst output buffer of samples
 * @param src input channels
 * @param channels number of channels (1 or 2)
 * @param aLength number of samples
 */
void virtch_deinterleave_float(float* dst, const float** src, int channels, size_t aLength)
{
#if (VMIX_SIMD == VMIX_SIMD_AVX || VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512 || VMIX_SIMD == VMIX_SIMD_SSE)
    _mm_prefetch((const char*)src[0], _MM_HINT_T0);
    if (channels == 2)
        _mm_prefetch((const char*)src[1], _MM_HINT_T0);
#endif
    if (channels == 1)
    {
        memcpy(dst, src[0], aLength * sizeof(float));
    }
    else if (channels == 2)
    {
        virtch_deinterleave_st_float(dst, src[0], src[1], aLength);
    }
}

/**
 * \brief Deinterleave an array of int16_t samples into two float arrays.
 * \param dst Output buffers for left and right channels.
 * \param src Input buffer of interleaved int16_t samples.
 * \param aLength Number of samples in the input buffer.
 */
void virtch_deinterleave_float_int16(float** dst, const int16_t* src, int aLength)
{
    for (int i = 0; i < aLength; i += 2)
    {
        dst[0][i >> 1] = src[i] * INV_ONE_16S;
        dst[1][i >> 1] = src[i + 1] * INV_ONE_16S;
    }
}
#ifdef TEST

/***
 Unit tests
 gcc virtch_simd.c -DTEST -o virtch -g -Os -mavx2
 cl /arch:AVX2 /O2 /Zi /DTEST virtch_simd.c /Fevirtch.exe
 ***/
#include <assert.h>
#include <stdio.h>

 // Timer-related macros
#ifdef __APPLE__
#include <mach/mach_time.h>
#define stop_watch_t uint64_t
#define begin_stopwatch() startTime = mach_absolute_time()
#define end_stopwatch() endTime = mach_absolute_time(); dt = endTime - startTime
#define LogV printf
#else 
#include <Windows.h>
#define stop_watch_t LARGE_INTEGER
#define begin_stopwatch() QueryPerformanceCounter(&startTime)
#define end_stopwatch() QueryPerformanceCounter(&endTime); dt = (int)(endTime.QuadPart - startTime.QuadPart)
#define LogV printf
#endif

static void* aligned_malloc( size_t size, int align )
{
    void* mem = malloc( size + (align-1) + sizeof(void*) );
    if (mem)
    {
        char* amem = ((char*)mem) + sizeof(void*);
        amem += align - ((uintptr_t)amem & (align - 1));
        ((void**)amem)[-1] = mem;
        return amem;
    }
    return mem;
}

static void aligned_free( void* mem )
{
    free( ((void**)mem)[-1] );
}

int main()
{
    stop_watch_t startTime;
    stop_watch_t endTime;
    int numSamples = 65536;
    int numChannels = 2;
    LogV("SIMD audio conversion test\n");
    LogV("# Test\t\t\t|Cycles (Ref.)\t\t|Cycles (SIMD)\t|Gain\n");
    sample_t* dst = (sample_t*) aligned_malloc(numSamples * 2 * numChannels, 64);
    int32_t* src = (int32_t*) aligned_malloc(numSamples * 4 * numChannels, 64);
    float* left = (float*)aligned_malloc(4096 * 4 * 2, 64);
    float* right = left + 4096;


    int dt = 0;
    int dtRef = 0;
    const int numLoop = 1000;
    for (int i = 0; i < numLoop; i++) {
        begin_stopwatch();
        virtch_downmix_32_16(src, dst, numSamples);
        end_stopwatch();
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        begin_stopwatch();
        virtch_downmix_32_16(src, dst, numSamples);
        end_stopwatch();
    }
    LogV("virtch_downmix_32_16\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt));

    int32_t vol[2] = {1<<VMIX_BITSHIFT, 1<<VMIX_BITSHIFT};
    virtch_set_features(0);
    for (int i = 0; i < numLoop; i++)
    {
        begin_stopwatch();
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << VMIX_FRACBITS, numSamples >> 1);
        end_stopwatch();
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++)
    {
        begin_stopwatch();
        virtch_mix_stereo((sample_t*)src, vol, (int32_t*)dst, 0, 1 << VMIX_FRACBITS, numSamples >> 1);
        end_stopwatch();
    }
    LogV("virtch_mix_stereo\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt));
    virtch_set_features(0);
    for (int i = 0; i < numLoop; i++)
    {
        begin_stopwatch();
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << VMIX_FRACBITS, numSamples >> 1);
        end_stopwatch();
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++)
    {
        begin_stopwatch();
        virtch_mix_stereo_st((sample_t*)src, vol, (int32_t*)dst, 0, 1 << VMIX_FRACBITS, numSamples >> 1);
        end_stopwatch();
    }
    LogV("virtch_mix_stereo_st\t\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt));
    int aLength = 4096;
    virtch_set_features(0);
    for (int i = 0; i < numLoop; i++) {
        begin_stopwatch();
        virtch_pack_float_int16_st(dst, left, right, aLength);
        end_stopwatch();
    }
    dtRef = dt;
    virtch_set_features(1);
    for (int i = 0; i < numLoop; i++) {
        begin_stopwatch();
        virtch_pack_float_int16_st(dst, left, right, aLength);
        end_stopwatch();
    }
    LogV("virtch_pack_float_int16_st\t|%d\t|%d\t|%d %%\n", dtRef, dt, (dtRef * 100 / dt));

    aligned_free(left);
	aligned_free(dst);
	aligned_free(src);
}
#endif
