#include <chrono>
#include <omp.h>

typedef union type_caster_union { /* a union between a float and an integer */
  public:
    float f;
    uint32_t i;
} placeholder_name;

#pragma omp declare target
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
#pragma omp end declare target

void log2_approx (
  std::vector<float> &inputs, 
  std::vector<float> &outputs,
  std::vector<int> &precision,
  const int num_inputs,
  const int precision_count,
  const int repeat)
{
  const int output_size = num_inputs * precision_count;

  float *d_inputs = inputs.data();
  float *d_outputs = outputs.data();

  #pragma omp target data map(to: d_inputs[0:num_inputs]) \
                          map(from: d_outputs[0:output_size])
  {
    for(int i = 0; i < precision_count; ++i) {

      auto start = std::chrono::high_resolution_clock::now(); 

      for (int k = 0; k < repeat; ++k) {
        const float p = precision[i];
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int j = 0; j < num_inputs; ++j) {
          d_outputs[i*num_inputs+j] = binary_log(d_inputs[j], p);
        }
      }

      auto end = std::chrono::high_resolution_clock::now(); 
      auto etime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::cout << "\nIterative approximation with " << precision[i] <<" bits of precision\n";
      std::cout << "Average kernel execution time " << etime * 1e-3 / repeat << " (us)\n";
    }
  }
}

