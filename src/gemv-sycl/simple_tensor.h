#ifndef SIMPLE_TENSOR_H_
#define SIMPLE_TENSOR_H_

#include <cassert>
#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

template <typename T>
class SimpleTensor {
 public:
  SimpleTensor(sycl::queue &q, unsigned height, unsigned width)
      : width_(width), height_(height), q_(q) {
    data_ = sycl::malloc_device<T>(height_ * width_, q_);
  }
  T* device_data() const { return data_; }
  /**
   * @brief generate a height_ * width_ matrix with random fp16 numbers
   */
  void reset();
  /**
   * @brief copy the numbers from device to the host
   */
  void to_host(T* host_data, unsigned n);
  /**
   * @brief move constructor
   */
  SimpleTensor(SimpleTensor&& other) noexcept
      : width_(other.width_), height_(other.height_), data_(other.data_) {
    other.data_ = nullptr;  // Ensure the other object won't delete the data
                            // after being destroyed
  }
  /**
   * @brief overload the assignment operator for move semantics
   */
  SimpleTensor& operator=(SimpleTensor&& other) noexcept {
    if (this != &other) {
      height_ = other.height_;
      width_ = other.width_;

      // Deallocate existing data
      sycl::free(data_, q_);

      // Take ownership of the new data
      data_ = other.data_;
      other.data_ = nullptr;
    }

    return *this;
  }
  ~SimpleTensor() { sycl::free(data_, q_); }

  unsigned int width_;
  unsigned int height_;
  // device data
  T* data_;
  sycl::queue q_;
};

template <typename T> void SimpleTensor<T>::reset() {
  unsigned int total_elements = height_ * width_;
  std::vector<T> rng (total_elements);
  std::mt19937 gen(19937);
  if constexpr (std::is_same<T, sycl::half>::value) {
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (unsigned n = 0; n < total_elements; ++n)
      rng[n] = sycl::vec<float, 1>(dis(gen))
                   .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    q_.memcpy(data_, rng.data(), total_elements * sizeof(T)).wait();
  } else if constexpr (std::is_same<T, int8_t>::value) {
    std::uniform_int_distribution<int> dis(-128, 127);
    for (unsigned n = 0; n < total_elements; ++n)
      rng[n] = dis(gen);
    q_.memcpy(data_, rng.data(), total_elements * sizeof(T)).wait();
  } else if constexpr (std::is_same<T, uint4_2>::value) {
    std::uniform_int_distribution<int> dis(0, 15);
    for (unsigned n = 0; n < total_elements; ++n) {
      rng[n].setX(dis(gen));
      rng[n].setY(dis(gen));
    }
    q_.memcpy(data_, rng.data(), total_elements * sizeof(T)).wait();
  }
}

template <typename T>
void SimpleTensor<T>::to_host(T* host_data, unsigned n) {
  unsigned int total_elements = height_ * width_;
  assert(n <= total_elements);
  q_.memcpy(host_data, data_, n * sizeof(T)).wait();
}

#endif  // SIMPLE_TENSOR_H_
