/*==============================================================================
 
 $Id$
 
 virtch_simd.h
 Audio processing and resampling from 32bit (int and float) to 16bit sound buffer
 Implementation for SSE, Altivec, ARM Neon, AVX
 
 ==============================================================================*/

#ifndef _VIRTCH_SIMD_H_
#define _VIRTCH_SIMD_H_


#if defined _MSC_VER
#define VMIX_FORCE_INLINE(type) static __inline type
#else
#define VMIX_FORCE_INLINE(type) static __inline type __attribute__((__always_inline__, __nodebug__)
#endif

// ARM Neon
#define VMIX_SIMD_NEON 1

// PowerPC
#define VMIX_SIMD_ALTIVEC 2

// SSE2 (2001)
#define VMIX_SIMD_SSE 3

// AVX : Same as SSE2 (no viable optimization possible)
#define VMIX_SIMD_AVX 4

// AVX2 : Haswell
#define VMIX_SIMD_AVX2 5

// AVX-512 Foundation (Knights Landing). Never tested.
#define VMIX_SIMD_AVX512 6

// Feature Conditional AVX
#define VMIX_FEATURE_AVX 0x1000

// Feature Conditional AVX
#define VMIX_FEATURE_AVX2 0x2000

// Feature Conditional AVX512
#define VMIX_FEATURE_AVX512 0x4000

// Auto-target
#if !defined VMIX_SIMD

#ifdef __ARM_NEON__
#define VMIX_SIMD VMIX_SIMD_NEON
#elif defined __AVX512F__
#define VMIX_SIMD VMIX_SIMD_AVX512
#elif defined __AVX2__
#define VMIX_SIMD VMIX_SIMD_AVX2
#elif defined __AVX__
#define VMIX_SIMD VMIX_SIMD_AVX
#elif defined __SSE2__ && !defined ANDROID
#define VMIX_SIMD VMIX_SIMD_SSE
#elif defined __ALTIVEC__
#define VMIX_SIMD VMIX_SIMD_ALTIVEC
#else
#define VMIX_SIMD 0
#endif

#endif

#include <stdlib.h>
#include <stdint.h>

#define SAMPLE_ALIGNED(ptr, stride)  (((size_t)(ptr) & ((1 << (stride)) - 1)) == 0)

#define sample_t int16_t
#define streamsample_t int32_t

#ifdef __arm__
#define SAMPLE_ALIGN 16
#else
#define SAMPLE_ALIGN 64
#endif


#if defined _M_ARM64
#include <arm64_neon.h>
typedef float32x4_t v4sf;  // vector of 4 float
typedef int32x4_t v4si;  // vector of 4 uint32

#elif defined __ARM_NEON__
#include <arm_neon.h>
typedef float32x4_t v4sf;  // vector of 4 float
typedef int32x4_t v4si;  // vector of 4 uint32

#elif defined ANDROID

#elif VMIX_SIMD == VMIX_SIMD_SSE || VMIX_SIMD == VMIX_SIMD_AVX ||  VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512

#if defined _MSC_VER

#if defined(_M_IX86) || defined(_M_X64)
#include <intrin.h>
#include <immintrin.h>
#endif
#else
#if defined(__x86_64__) || defined(__amd64__)
#include <immintrin.h>
#else
#error ("Invalid Target")
#endif

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


#if VMIX_SIMD == VMIX_SIMD_AVX2 || VMIX_SIMD == VMIX_SIMD_AVX512

VMIX_FORCE_INLINE(void)
unpackhi_epi16_store(__m256i* out, v8si a, v8si b)
{
	// Interleaves the upper 8 signed 16-bit integers in a with the upper 8 signed or unsigned 16-bit integers in b.
    _mm256_store_si256(out, _mm256_unpackhi_epi16(a, b));
}

VMIX_FORCE_INLINE(void)
unpacklo_epi16_store(__m256i* out, v8si a, v8si b)
{
    _mm256_store_si256(out, _mm256_unpacklo_epi16(a, b));
}


#elif defined __AVX__


#endif


#define VOL_MAX (1 << BITSHIFT)

#define BITSHIFT 9
#define FRACBITS 11
#define FRACMASK ((1<<FRACBITS)-1)

#ifdef __allegrex__
#define CLAMP(x, SAMPLE_MAX) __builtin_allegrex_min(__builtin_allegrex_max(x, SAMPLE_MAX - 1), SAMPLE_MAX)
#else
#define CLAMP(x, SAMPLE_MAX) ((x) > SAMPLE_MAX ? SAMPLE_MAX : ((x) < -SAMPLE_MAX - 1 ? - SAMPLE_MAX - 1 : (x)))
#endif

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
  
    /* enable SIMD processing. 0 to disable */
    void virtch_set_features(int features);
    
	/* downmix a 32bit buffer samples (sint32) to int16 */
    void virtch_downmix_32_16(const streamsample_t* src, int16_t *dst, size_t n);

	/* downmix a 32bit buffer samples (sint32) to int8 */
    void virtch_downmix_32_8(const streamsample_t* src, int8_t *dst, size_t n);

	/* downmix a 32bit buffer samples (sint32) fp32 */
    void virtch_downmix_32_fp32(const streamsample_t* src, float *dst, size_t n);
   
    /* mix mono stream with mono samples*/
	size_t virtch_mix_mono(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t n);
   
    /* mix stereo stream  with mono samples*/
	size_t virtch_mix_stereo(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t n);
   
    /* mix stereo stream with stereo samples*/
	size_t virtch_mix_stereo_st(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t n);
    
	/* convert int16 sample to floating buffer */
    void virtch_int16_to_fp(const sample_t* src, float* dst, int numChannels, size_t n);

	/* convert int8 sample to floating buffer */
    void virtch_int8_to_fp(const int8_t* src, float* dst, int numChannels, size_t n);
    
    /* convert float to int16 */
    int virtch_pack_float_int16(sample_t* outdata, const float** pcm, int channels, size_t n);

    /* deinterleave two arrays of float to a single array*/
    void virtch_deinterleave_float(float* outdata, const float** pcm, int channels, size_t n);
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

