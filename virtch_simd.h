// virtch_simd.h
// V0.8 2024-11-11
//
// Copyright (c) 2024 Stephane Denis
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

#ifndef VMIX_AUDIO_SIMD_H
#define VMIX_AUDIO_SIMD_H

#include <stdlib.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define VMIX_FORCE_INLINE(type) static __inline type
#else
#define VMIX_FORCE_INLINE(type) static inline type
#endif

#define EXPERIMENTAL_VIRTCH_NEON 1

// ARM Neon
#define VMIX_SIMD_NEON 1

// PowerPC
#define VMIX_SIMD_ALTIVEC 2

// SSE2 (2001)
#define VMIX_SIMD_SSE 3

// AVX 
#define VMIX_SIMD_AVX 4

// AVX2 : Haswell
#define VMIX_SIMD_AVX2 5

// AVX-512 Foundation (Knights Landing)
#define VMIX_SIMD_AVX512 6

// SIMD Feature Flags
#define VMIX_FEATURE_AVX      0x1000
#define VMIX_FEATURE_AVX2     0x2000
#define VMIX_FEATURE_AVX512   0x4000

// Default SIMD Detection
#ifndef VMIX_SIMD
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define VMIX_SIMD VMIX_SIMD_NEON
#elif defined(__AVX512F__)
#define VMIX_SIMD VMIX_SIMD_AVX512
#elif defined(__AVX2__)
#define VMIX_SIMD VMIX_SIMD_AVX2
#elif defined(__AVX__)
#define VMIX_SIMD VMIX_SIMD_AVX
#elif defined(__SSE2__) && !defined(__ANDROID__)
#define VMIX_SIMD VMIX_SIMD_SSE
#elif defined(__ALTIVEC__)
#define VMIX_SIMD VMIX_SIMD_ALTIVEC
#else
#define VMIX_SIMD VMIX_SIMD_NONE
#endif
#endif

// Type Definitions and Alignments
#define SAMPLE_ALIGNED(ptr, stride)  (((size_t)(ptr) & ((1 << (stride)) - 1)) == 0)
#define sample_t int16_t
#define streamsample_t int32_t

#if defined(__arm__)
#define SAMPLE_ALIGN 16
#else
#define SAMPLE_ALIGN 64
#endif

#if defined _MSC_VER
#define ALIGNED_AS(arg, al) arg
#else
#define ALIGNED_AS(arg, al) __builtin_assume_aligned(arg, al)
#endif

// SIMD Vector Types
#if VMIX_SIMD == VMIX_SIMD_NEON || defined(__ARM_NEON__)
#include <arm_neon.h>
typedef float32x4_t v4sf;
typedef int32x4_t v4si;
typedef int16x8_t v8sw;
typedef int16x4_t v4sw;
#define vec_set_4sw(a,b,c,d) {a, b, c, d}

#if !defined(__aarch64__) // ARM64 (ARMv8-A 64-bit)
static inline int16x8_t vqmovn_high_s32(int16x4_t low, int32x4_t high) {
    int16x4_t high_narrow = vqmovn_s32(high);
    return vcombine_s16(low, high_narrow);
}
#endif

#elif VMIX_SIMD == VMIX_SIMD_SSE || VMIX_SIMD == VMIX_SIMD_AVX || VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512
#include <immintrin.h>
typedef __m128 v4sf;
typedef __m128i v4si;

#ifdef __AVX__
typedef __m256 v8sf;
typedef __m256i v8si;
#define vec256_fmadd_16(a, b, c) _mm256_add_epi32(a, _mm256_mullo_epi16(b, c))
#endif

#ifdef __AVX512F__
typedef __m512 v16sf;
typedef __m512i v16si;
#define vec512_fmadd_16(a, b, c) _mm512_add_epi32(a, _mm512_mullo_epi16(b, c))
#endif

#define vec_setlo_4si(a, b, c, d)  _mm_set_epi16(0, d, 0, c, 0, b, 0, a)

#endif

// Helper Macros and Functions
#define VOL_MAX (1 << VMIX_BITSHIFT)
#define VMIX_BITSHIFT 9
#define VMIX_FRACBITS 11
#define VMIX_FP_SHIFT 4
#define VMIX_FRACMASK ((1 << VMIX_FRACBITS) - 1)
// Convert 32 to 16
#define BITSHIFT_VOL_32_16 (VMIX_BITSHIFT )
// Convert 32 to 8
#define BITSHIFT_VOL_32_8  (VMIX_BITSHIFT + 8)

#define CLAMP_SSE_INT(aValue, aMin, aMax) \
    _mm_max_epi32(_mm_min_epi32(aValue, _mm_set1_epi32(aMax)), _mm_set1_epi32(aMin))

#define CLAMP_SSE_FLOAT(aValue, aMin, aMax) \
    _mm_max_ps(_mm_min_ps(aValue, _mm_set1_ps(aMax)), _mm_set1_ps(aMin))

#define CLAMP_ALLEGREX_INT(aValue, aMax) __builtin_allegrex_min(__builtin_allegrex_max(aValue, aMax - 1), aMax)

#define CLAMP_STD(aValue, aMax) ((aValue) > aMax ? aMax : ((aValue) < -aMax - 1 ? -aMax - 1 : (aValue)))

#ifdef __allegrex__
#define CLAMP(aValue, aMax) CLAMP_ALLEGREX_INT(aValue, aMax)
#else
#define CLAMP(aValue, aMax) CLAMP_STD(aValue, aMax)
#endif
#define CLAMP_F(aValue) ((aValue) < -1.0f ? -1.0f : ((aValue) > 1.0f ? 1.0f : 0.0f))

// SIMD-Specific Functions
#if VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512
VMIX_FORCE_INLINE(void)
unpackhi_epi16_store(__m256i* out, v8si a, v8si b) {
    _mm256_store_si256(out, _mm256_unpackhi_epi16(a, b));
}

VMIX_FORCE_INLINE(void)
unpacklo_epi16_store(__m256i* out, v8si a, v8si b) {
    _mm256_store_si256(out, _mm256_unpacklo_epi16(a, b));
}
#endif


#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    /* enable SIMD processing. 0 to disable */
    void virtch_set_features(int features);
	/* downmix a 32bit buffer samples (sint32) to int16 */
    void virtch_downmix_32_16(const streamsample_t* src, int16_t *dst, size_t aLength);
	/* downmix a 32bit buffer samples (sint32) to int8 */
    void virtch_downmix_32_8(const streamsample_t* src, int8_t *dst, size_t aLength);
	/* downmix a 32bit buffer samples (sint32) fp32 */
    void virtch_downmix_32_fp32(const streamsample_t* src, float *dst, size_t aLength);
    /* mix mono stream with mono samples*/
	size_t virtch_mix_mono(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t aLength);
    /* mix stereo stream  with mono samples*/
	size_t virtch_mix_stereo(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t aLength);
    /* mix stereo stream with stereo samples*/
	size_t virtch_mix_stereo_st(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t aLength);
	/* convert int16 sample to floating buffer */
    void virtch_int16_to_fp(const sample_t* src, float* dst, int numChannels, size_t aLength);
	/* convert int8 sample to floating buffer */
    void virtch_int8_to_fp(const int8_t* src, float* dst, int numChannels, size_t aLength);
    /* convert float to int16 */
    size_t virtch_pack_float_int16(sample_t* outdata, const float** pcm, int channels, size_t aLength);
    size_t virtch_pack_float_int16_st(sample_t* __dst, const float* __left, const float* __right, size_t length);
    /* deinterleave two arrays of float to a single array*/
    void virtch_deinterleave_float(float* outdata, const float** pcm, int channels, size_t aLength);
    /* */
    void virtch_deinterleave_float_int16(float**dst, const int16_t* src,int length);
#ifdef __cplusplus
}
#endif /* __cplusplus */


#define FRACBITS VMIX_FRACBITS
#define FP_SHIFT VMIX_FP_SHIFT
#define BITSHIFT VMIX_BITSHIFT
#endif
