#include <chrono>
#include <sycl/sycl.hpp>

typedef union type_caster_union { /* a union between a float and an integer */
  public:
    float f;
    uint32_t i;
} placeholder_name;

float binary_log(float input, int precision)
{
  type_caster_union d1;
  d1.f = input;
  uint8_t exponent = ((d1.i & 0x7F800000) >> 23) - 127; // mask off the float's sign bit
  int m = 0;
  int sum_m = 0;
  float result = 0;
  int test = (1 << exponent);
  float y = input / test;
  bool max_condition_met = 0;
  uint64_t one = 1;
  uint64_t denom = 0;
  uint64_t prev_denom = 0;
  while((sum_m < precision + 1 && y != 1) || max_condition_met){
    m = 0;
    while((y < 2.f) && (sum_m + m < precision + 1)){
      y *= y;
      m++;
    }

    sum_m += m;
    prev_denom = denom;
    denom = one << sum_m;

    if(sum_m >= precision){ //break when we deliver as much precision as requested
      break;
    }
    if(prev_denom > denom){
      max_condition_met = 1;
      //std::cout << "Warning : unable to provide precision of 2^-" << precision <<
      //             " requested. Providing maximum precision of 2^-64" << std::endl;
      break;
    }

    result += 1.f / (float)denom;
    y /= 2.f;
  }
  return exponent + result;
}

void compute_log(sycl::nd_item<1> &item,
                       float* __restrict output,
                 const float* __restrict  input,
                 int r, int num_inputs, int precision)
{
  int i = item.get_global_id(0);
  if (i < num_inputs) {
    output[r*num_inputs+i] = binary_log(input[i], precision);
  }
}

void log2_approx (
  std::vector<float> &inputs,
  std::vector<float> &outputs,
  std::vector<int> &precision,
  const int num_inputs,
  const int precision_count,
  const int repeat)
{
  const int output_size = num_inputs * precision_count;
  std::vector<float> outputs_t (output_size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_inputs = sycl::malloc_device<float>(num_inputs, q);
  q.memcpy(d_inputs, inputs.data(), sizeof(float) * num_inputs);

  float *d_outputs = sycl::malloc_device<float>(output_size, q);

  sycl::range<1> gws ((num_inputs + 255)/256*256);
  sycl::range<1> lws (256);

  for(int i = 0; i < precision_count; ++i) {
    q.wait();
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < repeat; ++k) {
      q.submit([&] (sycl::handler &cgh) {
        const int p = precision[i];
        cgh.parallel_for<class approx>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          compute_log(item, d_outputs, d_inputs, i, num_inputs, p);
        });
      });
    }

    q.wait();
    auto end = std::chrono::high_resolution_clock::now();
    auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "\nIterative approximation with " << precision[i] <<" bits of precision\n";
    std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
  }

  q.memcpy(outputs.data(), d_outputs, sizeof(float) * output_size).wait();

  sycl::free(d_inputs, q);
  sycl::free(d_outputs, q);
}
