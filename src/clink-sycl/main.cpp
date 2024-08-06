#include <chrono>
#include <iostream>
#include <cstring>
#include <sycl/sycl.hpp>

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

// a multiple of WGS for simplicity
#define N 8192
#define WGS 256
#define SAMPLE_TEST_LEN 20000

float sigmoid(float x)
{
  return 1.f / (1.f + sycl::exp(-x));
}

#ifdef DEBUG
void dump (const char* work_path, const char* result_filename, const float* result) 
{
  char file_name[100];
  int i;

  FILE *fp;

  sprintf(file_name, "%s/%s", work_path, result_filename);
  // Open float_infer_result_1.hpp for output data write back.
  if (!(fp = fopen(file_name, "w"))) {
    printf("File %s cannot be opened for write.\n", result_filename);
    exit(-1);
  }
  for (i = 0; i < SAMPLE_TEST_LEN; ++i)
    fprintf(fp, "%f\n", result[i]);
  fclose(fp);
}
#endif

void init(const char* work_path, const char* input_filename, const char* weight_filename,
		float* sample_input, float* inW, float* intW, float* intB, float* outW, float* outB) 
{
  char file_name[100];

  float weightVal;

  int i, j, k;

  FILE *fp;

  sprintf(file_name, "%s/%s", work_path, input_filename);
  // Read in sample input from "input.hpp" file
  if (!(fp = fopen(file_name, "r"))) {
    printf("File %s cannot be opened for read.\n", input_filename);
    exit(-1);
  }

  for (i = 0; i < SAMPLE_TEST_LEN; ++i) {
    fscanf(fp, "%f", &sample_input[i]);
  }
  fclose(fp);

  // duplicate the sample using the first sample
  for (int i = 1; i < N; i++)
    memcpy(sample_input+i*SAMPLE_TEST_LEN, sample_input, SAMPLE_TEST_LEN*sizeof(float));

  // Load weights and perform inference for LSTM 1.
  sprintf(file_name, "%s/%s", work_path, weight_filename);
  if (!(fp = fopen(file_name, "r"))) {
    printf("File %s cannot be opened for read.\n", weight_filename);
    exit(-1);
  }
  for (j = 0; j < 4; ++j) {
    for (i = 0; i < 5; ++i) {
      fscanf(fp, "%f", &weightVal);
      inW[j*5+i] = weightVal;
    }
  }
  for (k = 0; k < 4; ++k) {
    for (j = 0; j < 5; ++j) {
      for (i = 0; i < 5; ++i) {
        fscanf(fp, "%f", &weightVal);
        intW[k*25+j*5+i] = weightVal;
      }
    }
  }
  for (j = 0; j < 4; ++j) {
    for (i = 0; i < 5; ++i) {
      fscanf(fp, "%f", &weightVal);
      intB[j*5+i] = weightVal;
    }
  }
  for (i = 0; i < 5; ++i) {
    fscanf(fp, "%f", &weightVal);
    outW[i] = weightVal;
  }
  fscanf(fp, "%f", &weightVal);
  *outB = weightVal;
  fclose(fp);
}

long lstm_n5( sycl::queue &q,
              const float* x, 
              const float* inW, 
              const float* intW, 
              const float* intB, 
              const float* outW, 
              const float* outB,
                    float* y) 
{
  float *d_x = sycl::malloc_device<float>(N*SAMPLE_TEST_LEN, q);
  q.memcpy(d_x, x, sizeof(float) * N * SAMPLE_TEST_LEN);

  float *d_inW = sycl::malloc_device<float>(20, q);
  q.memcpy(d_inW, inW, sizeof(float) * 20);

  float *d_intW = sycl::malloc_device<float>(100, q);
  q.memcpy(d_intW, intW, sizeof(float) * 100);

  float *d_intB = sycl::malloc_device<float>(20, q);
  q.memcpy(d_intB, intB, sizeof(float) * 20);

  float *d_outW = sycl::malloc_device<float>(5, q);
  q.memcpy(d_outW, outW, sizeof(float) * 5);

  float *d_outB = sycl::malloc_device<float>(1, q);
  q.memcpy(d_outB, outB, sizeof(float));

  float *d_y = sycl::malloc_device<float>(N*SAMPLE_TEST_LEN, q);
  
  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class lstm>(
      sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(WGS)),
      [=] (sycl::nd_item<1> item) {
      int t,i,j;
      int gid = item.get_global_id(0);
      
      float h_state[5] = {0,0,0,0,0};
      float c_state[5] = {0,0,0,0,0};
      float i_state[5] = {0,0,0,0,0};
      float f_state[5] = {0,0,0,0,0};
      float o_state[5] = {0,0,0,0,0};
      float g_state[5] = {0,0,0,0,0};
      
      for (t = 0; t < SAMPLE_TEST_LEN; ++t) {

        float x = ldg(&d_x[gid * SAMPLE_TEST_LEN + t]);

        for (j = 0; j < 5; ++j) {
          i_state[j] = ldg(&d_inW[j]) * x;
          for (i = 0; i < 5; ++i)
            i_state[j] += h_state[i] * ldg(&d_intW[j*5+i]);
          i_state[j] += ldg(&d_intB[j]);
          i_state[j] = sigmoid(i_state[j]);
        }
        
        for (j = 0; j < 5; ++j) {
          f_state[j] = ldg(&d_inW[5+j]) * x;
          for (i = 0; i < 5; ++i)
            f_state[j] += h_state[i] * ldg(&d_intW[25+j*5+i]);
          f_state[j] += ldg(&d_intB[5+j]);
          f_state[j] = sigmoid(f_state[j]);
        }

        for (j = 0; j < 5; ++j) {
          o_state[j] = ldg(&d_inW[10+j]) * x;
          for (i = 0; i < 5; ++i)
            o_state[j] += h_state[i] * ldg(&d_intW[50+j*5+i]);
          o_state[j] += ldg(&d_intB[10+j]);
          o_state[j] = sigmoid(o_state[j]);
        }

        for (j = 0; j < 5; ++j) {
          g_state[j] = ldg(&d_inW[15+j]) * x;
          for (i = 0; i < 5; ++i)
            g_state[j] += h_state[i] * ldg(&d_intW[75+j*5+i]);
          g_state[j] += ldg(&d_intB[15+j]);
          g_state[j] = sycl::tanh(g_state[j]);
        }

        for (j = 0; j < 5; ++j) {
          c_state[j] = c_state[j] * f_state[j] + g_state[j] * i_state[j];
          h_state[j] = sycl::tanh(c_state[j]) * o_state[j];
        }

        float y = ldg(&d_outB[0]);
        for (j = 0; j < 5; ++j)
          y += h_state[j] * ldg(&d_outW[j]);
        d_y[gid * SAMPLE_TEST_LEN + t] = y;
      }
    });
  }).wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(y, d_y, sizeof(float) * N * SAMPLE_TEST_LEN).wait();
  sycl::free(d_x ,q);
  sycl::free(d_inW ,q);
  sycl::free(d_intW ,q);
  sycl::free(d_intB ,q);
  sycl::free(d_outW ,q);
  sycl::free(d_outB ,q);
  sycl::free(d_y ,q);
  return time;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  float* sample_input = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);
  float* infer1_out = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);
  float* infer2_out = (float*) aligned_alloc(64, sizeof(float)*N*SAMPLE_TEST_LEN);

  float inW[20], intW[100], intB[20], outW[5];
  float outB;

  const char* work_path = "./";
  const char* input_filename = "input.hpp";
  const char* weight1_filename = "weight_1.hpp";
  const char* weight2_filename = "weight_2.hpp";
#ifdef DEBUG
  const char* result1_filename = "float_infer_result_1.hpp";
  const char* result2_filename = "float_infer_result_2.hpp";
#endif

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  long kernel_time = 0;
  for (int n = 0; n < repeat; n++) {
    init(work_path, input_filename, weight1_filename, sample_input, inW, intW, intB, outW, &outB) ;
    auto start = std::chrono::steady_clock::now();
    kernel_time += lstm_n5(q, sample_input, inW, intW, intB, outW, &outB, infer1_out);
    auto end = std::chrono::steady_clock::now();
    auto elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Device offload time: " <<  elapsedTime << " ms\n";

#ifdef DEBUG
    dump(work_path, result1_filename, infer1_out);
#endif

    init(work_path, input_filename, weight2_filename, sample_input, inW, intW, intB, outW, &outB) ;
    start = std::chrono::steady_clock::now();
    kernel_time += lstm_n5(q, sample_input, inW, intW, intB, outW, &outB, infer2_out);
    end = std::chrono::steady_clock::now();
    elapsedTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Device offload time: " <<  elapsedTime << " ms\n";

#ifdef DEBUG
    dump(work_path, result2_filename, infer2_out);
#endif
  }
  std::cout << "Average kernel time: " <<  kernel_time * 1e-6 / (2 * repeat) << " ms\n";

  free(sample_input);
  free(infer1_out);
  free(infer2_out);
  printf("Processing complete.\n");
  return 0;
}
