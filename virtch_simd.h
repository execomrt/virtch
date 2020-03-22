/*==============================================================================
 
 $Id$
 
 virtch_simd.h
 Audio processing and resampling from 32bit (int and float) to 16bit sound buffer
 Implementation for SSE, Altivec, ARM Neon, AVX
 
 ==============================================================================*/

#ifndef _VIRTCH_SIMD_H_
#define _VIRTCH_SIMD_H_

// ARM Neon
#define VMIX_SIMD_NEON 1

// PowerPC
#define VMIX_SIMD_ALTIVEC 2

// SSE2 (2001)
#define VMIX_SIMD_SSE 3

// AVX : Same as SSE2 (no viable optimization possible)
#define VMIX_SIMD_AVX 4

// AVX2 : Haswell. Never tested
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
#ifndef VMIX_SIMD

#ifdef __ARM_NEON__
#define VMIX_SIMD VMIX_SIMD_NEON
#elif defined __AVX512F__
#define VMIX_SIMD VMIX_SIMD_AVX512
#elif defined __AVX2__
#define VMIX_SIMD VMIX_SIMD_AVX2
#elif defined __AVX__
#define VMIX_SIMD VMIX_SIMD_AVX
#elif defined __SSE2__
#define VMIX_SIMD VMIX_SIMD_SSE
#elif defined __ALTIVEC__
#define VMIX_SIMD VMIX_SIMD_ALTIVEC
#else
#define VMIX_SIMD	0
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
    
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif

