#include <chrono>
#include "common.h"

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

void compute_log(nd_item<1> &item,
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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_inputs(inputs.data(), num_inputs);
  buffer<float, 1> d_outputs (output_size);

  range<1> gws ((num_inputs + 255)/256*256);
  range<1> lws (256);

  for(int i = 0; i < precision_count; ++i) {
    q.wait();
    auto start = std::chrono::high_resolution_clock::now(); 

    for (int k = 0; k < repeat; ++k) {
      q.submit([&] (handler &cgh) {
        const int p = precision[i];
        auto output = d_outputs.get_access<sycl_discard_write>(cgh); 
        auto input = d_inputs.get_access<sycl_read>(cgh); 
        cgh.parallel_for<class approx>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          compute_log(item, output.get_pointer(), input.get_pointer(),
                      i, num_inputs, p);
        });
      });
    }

    q.wait();
    auto end = std::chrono::high_resolution_clock::now(); 
    double etime = 
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    etime = (etime * 1e-9) / repeat;
    std::cout << "\nIterative approximation with " << precision[i] <<" bits of precision\n";
    std::cout << "Average kernel execution time " << etime << " (s)\n";
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_outputs.get_access<sycl_read>(cgh);
    cgh.copy(acc, outputs.data());
  }).wait();
}

