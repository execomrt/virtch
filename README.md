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



[virtch] Fast Audio sample conversion for 32bit integer/float, mono and stereo to 16bit
=====================================================================

Cross-platform open-source library
It reaches very fast processing speeds by utilizing SIMD functions, works for Android, iOS, Windows.
Can be used with following library : OpenSLES, libOGG, Mikmod
Support for pitch control (increment) and volume.

Test Units
=====================================================================
gcc virtch_simd.c -DTEST -o virtch -g -Ofast -msse4.2
Then type the generated executable
 ./virtch
 
====================
virtch_downmix_32_16 test... 
76305 cycles
12210 cycles (SIMD)
====================
virtch_mix_stereo test... 
247740 cycles
66036 cycles (SIMD)
====================
virtch_mix_stereo_st test... 
117185 cycles
72171 cycles (SIMD)



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
   
/* mix stereo stream with mono samples*/
size_t virtch_mix_stereo(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t n);
   
 /* mix stereo stream with stereo samples*/
size_t virtch_mix_stereo_st(const sample_t* src, const int32_t* volume, streamsample_t* dst, size_t offset, size_t increment, size_t n);
    
/* convert int16 sample to floating buffer */
void virtch_int16_to_fp(const sample_t* src, float* dst, int numChannels, size_t n);






