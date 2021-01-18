#include <chrono>
#include <iostream>
#include <cstring>
#include <hip/hip_runtime.h>

// a multiple of WGS for simplicity
#define N 8192
#define WGS 256
#define SAMPLE_TEST_LEN 20000

__device__
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

__global__ void
lstm_inference( const float *d_x, 
                const float *d_inW, 
                const float* d_intW, 
                const float* d_intB, 
                const float* d_outW, 
                const float* d_outB, 
                      float* d_y)
{

  int t,i,j;
  int gid = blockDim.x * blockIdx.x + threadIdx.x;

  float h_state[5] = {0,0,0,0,0};
  float c_state[5] = {0,0,0,0,0};
  float i_state[5] = {0,0,0,0,0};
  float f_state[5] = {0,0,0,0,0};
  float o_state[5] = {0,0,0,0,0};
  float g_state[5] = {0,0,0,0,0};

  for (t = 0; t < SAMPLE_TEST_LEN; ++t) {
    for (j = 0; j < 5; ++j) {
      i_state[j] = d_inW[j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        i_state[j] += h_state[i] * d_intW[j*5+i];
      i_state[j] += d_intB[j];
      i_state[j] = sigmoid(i_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      f_state[j] = d_inW[5+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        f_state[j] += h_state[i] * d_intW[25+j*5+i];
      f_state[j] += d_intB[5+j];
      f_state[j] = sigmoid(f_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      o_state[j] = d_inW[10+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        o_state[j] += h_state[i] * d_intW[50+j*5+i];
      o_state[j] += d_intB[10+j];
      o_state[j] = sigmoid(o_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      g_state[j] = d_inW[15+j] * d_x[gid * SAMPLE_TEST_LEN + t];
      for (i = 0; i < 5; ++i)
        g_state[j] += h_state[i] * d_intW[75+j*5+i];
      g_state[j] += d_intB[15+j];
      g_state[j] = tanh(g_state[j]);
    }

    for (j = 0; j < 5; ++j) {
      c_state[j] = c_state[j] * f_state[j] + g_state[j] * i_state[j];
      h_state[j] = tanh(c_state[j]) * o_state[j];
    }

    d_y[gid * SAMPLE_TEST_LEN + t] = d_outB[0];
    for (j = 0; j < 5; ++j)
      d_y[gid * SAMPLE_TEST_LEN + t] += h_state[j] * d_outW[j];
  }
}

#ifdef DEBUG
void dump (const char* work_path, const char* result_filename, const float* result) {
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
		float* sample_input, float* inW, float* intW, float* intB, float* outW, float* outB) {

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

void  lstm_n5(
		const float* x, 
		const float* inW, 
		const float* intW, 
		const float* intB, 
		const float* outW, 
		const float* outB,
	       	float* y) {

  float *d_x, *d_inW, *d_intW, *d_intB, *d_outW, *d_y, *d_outB;
  hipMalloc((void**)&d_x, N * SAMPLE_TEST_LEN * sizeof(float));
  hipMalloc((void**)&d_inW, 20 * sizeof(float));
  hipMalloc((void**)&d_intW, 100 * sizeof(float));
  hipMalloc((void**)&d_intB, 20 * sizeof(float));
  hipMalloc((void**)&d_outW, 5 * sizeof(float));
  hipMalloc((void**)&d_outB, sizeof(float));
  hipMalloc((void**)&d_y, N * SAMPLE_TEST_LEN * sizeof(float));

  hipMemcpy(d_x, x, N * SAMPLE_TEST_LEN * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_inW, inW, 20 * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_intW, intW, 100 * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_intB, intB, 20 * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_outW, outW, 5 * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_outB, outB, 1 * sizeof(float), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(lstm_inference, N/WGS, WGS, 0, 0, 
      d_x, d_inW, d_intW, d_intB, d_outW, d_outB, d_y);

  hipMemcpy(y, d_y, N * SAMPLE_TEST_LEN * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_x);
  hipFree(d_inW);
  hipFree(d_intW);
  hipFree(d_intB);
  hipFree(d_outW);
  hipFree(d_outB);
  hipFree(d_y);
}

int main() {

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
    const char* result1_filename = "cuda_float_infer_result_1.hpp";
    const char* result2_filename = "cuda_float_infer_result_2.hpp";
#endif

    for (int n = 0; n < 10; n++) {
      init(work_path, input_filename, weight1_filename, sample_input, inW, intW, intB, outW, &outB) ;
      auto start = std::chrono::steady_clock::now();
      lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer1_out);
      auto end = std::chrono::steady_clock::now();
      auto elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
      std::cout << "Execute time: " <<  elapsedTime << " ms\n";
	
#ifdef DEBUG
      dump(work_path, result1_filename, infer1_out);
#endif


      init(work_path, input_filename, weight2_filename, sample_input, inW, intW, intB, outW, &outB) ;
      start = std::chrono::steady_clock::now();
      lstm_n5(sample_input, inW, intW, intB, outW, &outB, infer2_out);
      end = std::chrono::steady_clock::now();
      elapsedTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
      std::cout << "Execute time: " <<  elapsedTime << " ms\n";

#ifdef DEBUG
      dump(work_path, result2_filename, infer2_out);
#endif
    }

    free(sample_input);
    free(infer1_out);
    free(infer2_out);
    printf("Processing complete.\n");
    return 0;
}

