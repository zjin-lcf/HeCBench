#pragma once
 
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomicAdd(T *addr, const T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomicAdd(T1 *addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
  return atm.fetch_add(operand);
}

template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomicAdd(T &addr, const T operand) {
  auto atm =
      sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr);
  return atm.fetch_add(operand);
}

template <sycl::access::address_space addressSpace =
              sycl::access::address_space::generic_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device,
          typename T1, typename T2>
inline T1 atomicAdd(T1 &addr, T2 operand) {
  auto atm =
      sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr);
  return atm.fetch_add(operand);
}
