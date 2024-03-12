#pragma once 
// A fixed-size array type usable from both host and
// device code.

template <typename T, int size_>
struct Array {
  T data[size_];

  T operator[](int i) const {
    return data[i];
  }
  T& operator[](int i) {
    return data[i];
  }
#if defined(USE_ROCM)
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  static constexpr int size(){return size_;}
  // Fill the array with x.
  Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};
