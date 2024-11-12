# Audio Processing with SIMD Optimization
by Stephane Denis

This project provides optimized C implementations of audio processing functions, leveraging SIMD (Single Instruction, Multiple Data) instructions (AVX2, SSE, and NEON) to enhance performance on various hardware architectures, including x86 (Windows and macOS) and ARM (e.g., Apple Silicon).

The SIMD versions of these functions have shown significant performance improvements, ideal for applications requiring high-speed audio processing with minimal latency.

## Table of Contents
- [Overview](#overview)
- [Functions](#functions)
  - [`virtch_downmix_32_16`](#virtch_downmix_32_16)
  - [`virtch_mix_stereo`](#virtch_mix_stereo)
  - [`virtch_mix_stereo_st`](#virtch_mix_stereo_st)
  - [`virtch_pack_float_int16_st`](#virtch_pack_float_int16_st)
- [SIMD Implementations](#simd-implementations)
- [Compilation](#compilation)
- [Performance](#performance)
- [License](#license)

## Overview

This project contains several audio processing functions that perform downmixing, mixing, and packing tasks:
- **Downmixing** from 32-bit float to 16-bit integer audio.
- **Mixing** stereo audio with volume adjustments.
- **Packing** floating-point samples into 16-bit signed integer format.

These functions utilize SIMD instructions for significant performance enhancements on compatible hardware.

## Functions

### `virtch_downmix_32_16`
Downmixes a 32-bit floating-point stereo audio buffer to a 16-bit signed integer buffer.

- **Parameters**:
  - `src` (input): Pointer to the source audio buffer (32-bit float).
  - `dst` (output): Pointer to the destination buffer (16-bit int).
  - `length`: Number of samples to process.
- **Description**: The function reduces bit depth and downmixes the audio data, processing multiple samples at once via SIMD.

### `virtch_mix_stereo`
Mixes a stereo audio source buffer into a 32-bit destination buffer, applying volume levels and pitch adjustments.

- **Parameters**:
  - `src` (input): Pointer to the source audio buffer.
  - `vol`: Volume levels for left and right channels.
  - `dst` (output): Pointer to the destination buffer (32-bit).
  - `offset`: Initial sample offset.
  - `increment`: Pitch increment for source buffer reading.
  - `length`: Number of stereo samples to process.
- **Description**: The function mixes the input audio with specified volume settings, adjusting the pitch by modifying the offset on each iteration. SIMD versions (SSE, AVX2, NEON) process multiple samples per loop for higher efficiency.

### `virtch_mix_stereo_st`
Similar to `virtch_mix_stereo`, but processes stereo samples and applies separate processing for the left and right channels.

- **Parameters**:
  - Same as `virtch_mix_stereo`.
- **Description**: Designed for specific stereo processing needs, this function handles left and right channels independently within each SIMD operation, optimizing for cases with separate channel processing requirements.

### `virtch_pack_float_int16_st`
Packs floating-point samples into 16-bit signed integer samples for storage or further processing.

- **Parameters**:
  - `src`: Pointer to the floating-point source buffer.
  - `dst`: Pointer to the destination buffer (16-bit int).
  - `length`: Number of samples to process.
- **Description**: Converts floating-point audio samples to 16-bit signed integer format using SIMD instructions for faster processing.

## SIMD Implementations

Each function supports SIMD optimizations with multiple implementations:
- **AVX2** (x86, Windows and macOS): 256-bit processing for high-speed data handling.
- **SSE** (x86, fallback): 128-bit processing for older CPUs.
- **NEON** (ARM, Apple Silicon): 128-bit processing for ARM-based architectures.

The appropriate SIMD implementation is selected based on available hardware features.

## Compilation

### macOS (Apple Silicon and Intel)
Use `gcc`:
```sh
gcc virtch_simd.c -DTEST -o virtch -g -Os
```

### Windows (x86, Visual Studio 2022)
1. Create a new Visual Studio project.
2. Add `virtch_simd.c` to your project.
3. Ensure SIMD instructions are enabled by adding appropriate flags to the compiler (e.g., `/arch:AVX2` for AVX2).

### Linux
To compile on Linux, ensure `gcc` is installed and use:
```sh
gcc virtch_simd.c -DTEST -o virtch -g -Os
```

## Performance

The following table summarizes performance improvements observed with SIMD optimizations:

| Test                        | Cycles (Ref.) | Cycles (SIMD) | Gain  |
|-----------------------------|---------------|---------------|-------|
| `virtch_downmix_32_16`      | 401           | 32            | 1253% |
| `virtch_mix_stereo`         | 347           | 176           | 197%  |
| `virtch_mix_stereo_st`      | 402           | 324           | 124%  |
| `virtch_pack_float_int16_st` | 100           | 70            | 142%  |

These gains reflect the effectiveness of SIMD for high-performance audio processing tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 

Let me know if you'd like further customization for specific platforms, usage examples, or additional detail in any section!