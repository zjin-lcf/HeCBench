#pragma once

template<typename T, typename sycl::memory_scope MemoryScope = sycl::memory_scope::work_group>
static inline void atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::generic_space> ref(val);
  ref.fetch_add(delta);
}

