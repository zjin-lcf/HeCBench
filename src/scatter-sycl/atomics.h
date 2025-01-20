#pragma once

template<typename T, typename V>
inline T atomicCAS(T *val, T expected, V desired)
{
  T expected_value = expected;
  auto atm = sycl::atomic_ref<T,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  atm.compare_exchange_strong(expected_value, (T)desired);
  return expected_value;
}

#define ATOMIC(NAME)                                                                                                                  \
  template <typename scalar, size_t size> struct Atomic##NAME##IntegerImpl;                                                           \
                                                                                                                                      \
  template <typename scalar> struct Atomic##NAME##IntegerImpl<scalar, 4> {                                                            \
    inline void operator()(scalar *address, scalar val) {                                                                             \
      uint32_t *address_as_ui = (uint32_t *)address;                                                                                  \
      uint32_t old = *address_as_ui;                                                                                                  \
      uint32_t assumed;                                                                                                               \
                                                                                                                                      \
      do {                                                                                                                            \
        assumed = old;                                                                                                                \
        old = atomicCAS(address_as_ui, assumed, OP(val, (scalar)old));                                                                \
      } while (assumed != old);                                                                                                       \
    }                                                                                                                                 \
  };                                                                                                                                  \
                                                                                                                                      \
  template <typename scalar> struct Atomic##NAME##IntegerImpl<scalar, 8> {                                                            \
    inline void operator()(scalar *address, scalar val) {                                                                             \
      unsigned long long *address_as_ull = (unsigned long long *)address;                                                             \
      unsigned long long old = *address_as_ull;                                                                                       \
      unsigned long long assumed;                                                                                                     \
                                                                                                                                      \
      do {                                                                                                                            \
        assumed = old;                                                                                                                \
        old = atomicCAS(address_as_ull, assumed, OP(val, (scalar)old));                                                               \
      } while (assumed != old);                                                                                                       \
    }                                                                                                                                 \
  };                                                                                                                                  \
                                                                                                                                      \
  template <typename scalar, size_t size> struct Atomic##NAME##DecimalImpl;                                                           \
                                                                                                                                      \
  template <typename scalar> struct Atomic##NAME##DecimalImpl<scalar, 4> {                                                            \
    inline void operator()(scalar *address, scalar val) {                                                                             \
      int *address_as_i = (int *)address;                                                                                             \
      int old = *address_as_i;                                                                                                        \
      int assumed;                                                                                                                    \
                                                                                                                                      \
      do {                                                                                                                            \
        assumed = old;                                                                                                                \
        old = atomicCAS(address_as_i, assumed,                                                                                        \
                        sycl::bit_cast<int, float>(OP(val, sycl::bit_cast<float>(assumed))));                                         \
      } while (assumed != old);                                                                                                       \
    }                                                                                                                                 \
  };                                                                                                                                  \
                                                                                                                                      \
  template <typename scalar> struct Atomic##NAME##DecimalImpl<scalar, 8> {                                                            \
    inline void operator()(scalar *address, scalar val) {                                                                             \
      unsigned long long int *address_as_ull =                                                                                        \
          (unsigned long long int *)address;                                                                                          \
      unsigned long long int old = *address_as_ull;                                                                                   \
      unsigned long long int assumed;                                                                                                 \
                                                                                                                                      \
      do {                                                                                                                            \
        assumed = old;                                                                                                                \
        old = atomicCAS(                                                                                                              \
            address_as_ull, assumed,                                                                                                  \
            sycl::bit_cast<long long, double>(OP(val, sycl::bit_cast<double>(assumed))));                                             \
      } while (assumed != old);                                                                                                       \
    }                                                                                                                                 \
  };

template<typename T>
inline T atomicAdd(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_add(val);
}

#define OP(X, Y) Y *X
ATOMIC(Mul)
#undef OP
static inline void atomicMul(int32_t *address, int32_t val) {
  AtomicMulIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline void atomicMul(int64_t *address, int64_t val) {
  AtomicMulIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline void atomicMul(float *address, float val) {
  AtomicMulDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline void atomicMul(double *address, double val) {
  AtomicMulDecimalImpl<double, sizeof(double)>()(address, val);
}

#define OP(X, Y) Y / X
ATOMIC(Div)
#undef OP
static inline void atomicDiv(int32_t *address, int32_t val) {
  AtomicDivIntegerImpl<int32_t, sizeof(int32_t)>()(address, val);
}
static inline void atomicDiv(int64_t *address, int64_t val) {
  AtomicDivIntegerImpl<int64_t, sizeof(int64_t)>()(address, val);
}
static inline void atomicDiv(float *address, float val) {
  AtomicDivDecimalImpl<float, sizeof(float)>()(address, val);
}
static inline void atomicDiv(double *address, double val) {
  AtomicDivDecimalImpl<double, sizeof(double)>()(address, val);
}

template<typename T>
inline T atomicMin(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_min(val);
}

template<typename T>
inline T atomicMax(T *address, T val)
{
  sycl::atomic_ref<T, 
                   sycl::memory_order::relaxed,
                   sycl::memory_scope::device,
                   sycl::access::address_space::global_space> ao (*address);
  return ao.fetch_max(val);
}

