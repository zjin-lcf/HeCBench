#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils.h"

template <typename T>
void toGPU(T* const output, T const* const input, size_t const num)
{
  CUDA_RT_CALL(
      dpct::get_default_queue().memcpy(output, input, num * sizeof(T)).wait());
}

template <typename T>
void fromGPU(T* const output, T const* const input, size_t const num)
{
  CUDA_RT_CALL(
      dpct::get_default_queue().memcpy(output, input, num * sizeof(T)).wait());
}

template <>
void fromGPU<void>(void* const output, void const* const input, size_t const num)
{
  CUDA_RT_CALL(dpct::get_default_queue().memcpy(output, input, num).wait());
}

template <typename T>
void runBitPackingOnGPU(
    T const* const inputHost,
    void* const outputHost,
    int const numBitsMax,
    size_t const n,
    int* const numBitsOut,
    T* const minValOut)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  T* input;

  CUDA_RT_CALL(input = (T *)sycl::malloc_device(n * sizeof(T), q_ct1));
  toGPU(input, inputHost, n);

  void* output;
  size_t const packedSize = (((numBitsMax * n) / 64U) + 1U) * 8U;

  size_t* numDevice;
  CUDA_RT_CALL(numDevice =
                   (size_t *)sycl::malloc_device(sizeof(numDevice), q_ct1));
  toGPU(numDevice, &n, 1);

  CUDA_RT_CALL(output = (void *)sycl::malloc_device(packedSize, q_ct1));
  CUDA_RT_CALL(q_ct1.memset(output, 0, packedSize).wait());

  T* minValueDevice;
  CUDA_RT_CALL(minValueDevice =
                   (T *)sycl::malloc_device(sizeof(*minValueDevice), q_ct1));
  unsigned char* numBitsDevice;
  CUDA_RT_CALL(numBitsDevice = (unsigned char *)sycl::malloc_device(
                   sizeof(*numBitsDevice), q_ct1));

  void* workspace;
  size_t workspaceBytes = requiredWorkspaceSize(n, TypeOf<T>());
  CUDA_RT_CALL(workspace = (void *)sycl::malloc_device(workspaceBytes, q_ct1));

  const nvcompType_t inType = TypeOf<T>();

  compress(
      workspace,
      workspaceBytes,
      inType,
      output,
      input,
      numDevice,
      n,
      minValueDevice,
      numBitsDevice);

  fromGPU(minValOut, minValueDevice, 1);

  unsigned char numBits;
  fromGPU(&numBits, numBitsDevice, 1);
  *numBitsOut = numBits;

  fromGPU(outputHost, output, std::min(packedSize, n * sizeof(T)));

  CUDA_RT_CALL(sycl::free(input, q_ct1));
  CUDA_RT_CALL(sycl::free(output, q_ct1));
  CUDA_RT_CALL(sycl::free(workspace, q_ct1));
  CUDA_RT_CALL(sycl::free(minValueDevice, q_ct1));
  CUDA_RT_CALL(sycl::free(numBitsDevice, q_ct1));
}

int main() {
 int const offset = 87231;
  int const numBits = 13;

  // unpack doesn't handle 0 bits
  std::vector<size_t> const sizes{2, 123, 3411, 83621, 872163, 100000001};

  using T = int32_t;

  // generate a variety of random numbers
  std::vector<T> source(sizes.back());
  std::srand(0);
  for (T& v : source) {
    v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();
  }

  size_t const numBytes = sizes.back() * sizeof(T);
  T* inputHost = (T*) aligned_alloc (1024, numBytes);
  void* outputHost = aligned_alloc (1024, numBytes);

  for (size_t const n : sizes) {
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1U << numBits) - 1)) + offset;
    }

    T minValue;
    int numBitsAct;
    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

    assert(numBitsAct <= numBits);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    assert(unpackedHost.size() == n);

    bool ok = true;
    // checking 100 million entries can take a while, so sample instead
    size_t const numSamples = static_cast<size_t>(std::sqrt(n)) + 1;
    for (size_t i = 0; i < numSamples; ++i) {
      // only works for arrays less than 4 Billion.
      size_t const idx = static_cast<uint32_t>(source[i]) % n;
      if (unpackedHost[idx] != inputHost[idx]) {
        ok = false;
        break;
      }
    }
    printf("n = %zu: %s\n", n, ok ? "PASS" : "FAILED");  
  }

  free(inputHost);
  free(outputHost);
}
