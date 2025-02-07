#pragma once

#define ATOMIC(NAME)                                                           \
  template <typename scalar, size_t size> struct Atomic##NAME##IntegerImpl;    \
                                                                               \
  template <typename scalar> struct Atomic##NAME##IntegerImpl<scalar, 4> {     \
    inline __device__ void operator()(scalar *address, scalar val) {           \
      uint32_t *address_as_ui = (uint32_t *)address;                           \
      uint32_t old = *address_as_ui;                                           \
      uint32_t assumed;                                                        \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old = atomicCAS(address_as_ui, assumed, OP(val, (scalar)old));         \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename scalar> struct Atomic##NAME##IntegerImpl<scalar, 8> {     \
    inline __device__ void operator()(scalar *address, scalar val) {           \
      unsigned long long *address_as_ull = (unsigned long long *)address;      \
      unsigned long long old = *address_as_ull;                                \
      unsigned long long assumed;                                              \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old = atomicCAS(address_as_ull, assumed, OP(val, (scalar)old));        \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename scalar, size_t size> struct Atomic##NAME##DecimalImpl;    \
                                                                               \
                                                                               \
  template <typename scalar> struct Atomic##NAME##DecimalImpl<scalar, 4> {     \
    inline __device__ void operator()(scalar *address, scalar val) {           \
      int *address_as_i = (int *)address;                                      \
      int old = *address_as_i;                                                 \
      int assumed;                                                             \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old = atomicCAS(address_as_i, assumed,                                 \
                        __float_as_int(OP(val, __int_as_float(assumed))));     \
      } while (assumed != old);                                                \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <typename scalar> struct Atomic##NAME##DecimalImpl<scalar, 8> {     \
    inline __device__ void operator()(scalar *address, scalar val) {           \
      unsigned long long int *address_as_ull =                                 \
          (unsigned long long int *)address;                                   \
      unsigned long long int old = *address_as_ull;                            \
      unsigned long long int assumed;                                          \
                                                                               \
      do {                                                                     \
        assumed = old;                                                         \
        old = atomicCAS(                                                       \
            address_as_ull, assumed,                                           \
            __double_as_longlong(OP(val, __longlong_as_double(assumed))));     \
      } while (assumed != old);                                                \
    }                                                                          \
  };

#define OP(X, Y) Y + X
ATOMIC(Add)
#undef OP
static inline __device__ void atomAdd(int32_t *address, int32_t val) {
  atomicAdd(address, val);
}
static inline __device__ void atomAdd(int64_t *address, int64_t val) {
  AtomicAddIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ void atomAdd(float *address, float val) {
  atomicAdd(address, val);
}
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000)
static inline __device__ void atomAdd(double *address, double val) {
  AtomicAddDecimalImpl<double, sizeof(double)>()(address, val);
}
#else
static inline __device__ void atomAdd(double *address, double val) {
  atomicAdd(address, val);
}
#endif

#define OP(X, Y) Y *X
ATOMIC(Mul)
#undef OP
static inline __device__ void atomMul(int32_t *address, int32_t val) {
  AtomicMulIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline __device__ void atomMul(int64_t *address, int64_t val) {
  AtomicMulIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ void atomMul(float *address, float val) {
  AtomicMulDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMul(double *address, double val) {
  AtomicMulDecimalImpl<double, sizeof(double)>()(address, val);
}

#define OP(X, Y) Y / X
ATOMIC(Div)
#undef OP
static inline __device__ void atomDiv(int32_t *address, int32_t val) {
  AtomicDivIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline __device__ void atomDiv(int64_t *address, int64_t val) {
  AtomicDivIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ void atomDiv(float *address, float val) {
  AtomicDivDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomDiv(double *address, double val) {
  AtomicDivDecimalImpl<double, sizeof(double)>()(address, val);
}

#define OP(X, Y) max(Y, X)
ATOMIC(Max)
#undef OP
static inline __device__ void atomMax(int32_t *address, int32_t val) {
  atomicMax(address, val);
}
static inline __device__ void atomMax(int64_t *address, int64_t val) {
  AtomicMaxIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ void atomMax(float *address, float val) {
  AtomicMaxDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMax(double *address, double val) {
  AtomicMaxDecimalImpl<double, sizeof(double)>()(address, val);
}

#define OP(X, Y) min(Y, X)
ATOMIC(Min)
#undef OP
static inline __device__ void atomMin(int32_t *address, int32_t val) {
  atomicMin(address, val);
}
static inline __device__ void atomMin(int64_t *address, int64_t val) {
  AtomicMinIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline __device__ void atomMin(float *address, float val) {
  AtomicMinDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline __device__ void atomMin(double *address, double val) {
  AtomicMinDecimalImpl<double, sizeof(double)>()(address, val);
}
