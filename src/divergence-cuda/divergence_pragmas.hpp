#ifndef DIVERGENCE_PRAGMAS_HPP
#define DIVERGENCE_PRAGMAS_HPP

#if defined(__INTEL_COMPILER)
#define NOVECDEP _Pragma("ivdep")
#define ALWAYSVECTORIZE _Pragma("vector always")
#define ALIGN(vardec) __declspec(align) vardec
#define ALIGNTO(vardec, boundary) \
  __declspec(align(boundary)) vardec
#elif defined(__clang__)
/* Clang (including hipcc and Clang CUDA) defines __GNUG__ == 4 for
   compatibility, which would otherwise look like G++ < 4.9. */
#define NOVECDEP _Pragma("clang loop vectorize(assume_safety)")
#define ALWAYSVECTORIZE _Pragma("clang loop vectorize(enable)")
#define ALIGN(vardec) __attribute__((aligned)) vardec
#define ALIGNTO(vardec, boundary) \
  __attribute__((aligned(boundary))) vardec
#elif defined(__GNUG__)
#if(__GNUG__ == 4 && __GNUC_MINOR__ >= 9) || __GNUG__ > 4
#define NOVECDEP _Pragma("GCC ivdep")
#define ALWAYSVECTORIZE _Pragma("GCC vector always")
#else
#pragma message( \
    "G++ <4.9 Does not support vectorization pragmas")
#define NOVECDEP
#define ALWAYSVECTORIZE
#endif

#define ALIGN(vardec) __attribute__((aligned)) vardec
#define ALIGNTO(vardec, boundary) \
  __attribute__((aligned(boundary))) vardec
#endif

#endif
