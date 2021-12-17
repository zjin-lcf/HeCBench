#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "kernels.h"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input = Layer(0, 0, 28*28);
static Layer l_c1 = Layer(5*5, 6, 24*24*6);
static Layer l_s1 = Layer(4*4, 1, 6*6*6);
static Layer l_f = Layer(6*6*6, 10, 10);

static void learn(int iter);
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

static inline void loaddata()
{
  mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
      &train_set, &train_cnt);
  mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
      &test_set, &test_cnt);
}

// replace cublas function in the case n = 10
void snrm2(const int n, const float *x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));
  hipMemcpy(r, x, n * sizeof(float), hipMemcpyDeviceToHost);
  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

int main(int argc, const  char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iter = atoi(argv[1]);
  srand(123);
  loaddata();
  learn(iter);
  test();
  return 0;
}

// Forward propagation of a single row in dataset
static double forward_pass(double data[28][28])
{
  float input[28][28];

  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }

  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  clock_t start, end;
  start = clock();

  l_input.setOutput((float *)input);

  hipLaunchKernelGGL(fp_preact_c1, dim3(64), dim3(64), 0, 0, (float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  hipLaunchKernelGGL(fp_bias_c1, dim3(64), dim3(64), 0, 0, (float (*)[24][24])l_c1.preact, l_c1.bias);
  hipLaunchKernelGGL(apply_step_function, dim3(64), dim3(64), 0, 0, l_c1.preact, l_c1.output, l_c1.O);

  hipLaunchKernelGGL(fp_preact_s1, dim3(64), dim3(64), 0, 0, (float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  hipLaunchKernelGGL(fp_bias_s1, dim3(64), dim3(64), 0, 0, (float (*)[6][6])l_s1.preact, l_s1.bias);
  hipLaunchKernelGGL(apply_step_function, dim3(64), dim3(64), 0, 0, l_s1.preact, l_s1.output, l_s1.O);

  hipLaunchKernelGGL(fp_preact_f, dim3(64), dim3(64), 0, 0, (float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  hipLaunchKernelGGL(fp_bias_f, dim3(64), dim3(64), 0, 0, l_f.preact, l_f.bias);
  hipLaunchKernelGGL(apply_step_function, dim3(64), dim3(64), 0, 0, l_f.preact, l_f.output, l_f.O);

  end = clock();
  return ((double) (end - start)) / CLOCKS_PER_SEC;
}

// Back propagation to update weights
static double back_pass()
{
  clock_t start, end;

  start = clock();

  hipLaunchKernelGGL(bp_weight_f, dim3(64), dim3(64), 0, 0, (float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  hipLaunchKernelGGL(bp_bias_f, dim3(64), dim3(64), 0, 0, l_f.bias, l_f.d_preact);

  hipLaunchKernelGGL(bp_output_s1, dim3(64), dim3(64), 0, 0, (float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  hipLaunchKernelGGL(bp_preact_s1, dim3(64), dim3(64), 0, 0, (float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  hipLaunchKernelGGL(bp_weight_s1, dim3(64), dim3(64), 0, 0, (float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  hipLaunchKernelGGL(bp_bias_s1, dim3(64), dim3(64), 0, 0, l_s1.bias, (float (*)[6][6])l_s1.d_preact);

  hipLaunchKernelGGL(bp_output_c1, dim3(64), dim3(64), 0, 0, (float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  hipLaunchKernelGGL(bp_preact_c1, dim3(64), dim3(64), 0, 0, (float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  hipLaunchKernelGGL(bp_weight_c1, dim3(64), dim3(64), 0, 0, (float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  hipLaunchKernelGGL(bp_bias_c1, dim3(64), dim3(64), 0, 0, l_c1.bias, (float (*)[24][24])l_c1.d_preact);


  hipLaunchKernelGGL(apply_grad, dim3(64), dim3(64), 0, 0, l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  hipLaunchKernelGGL(apply_grad, dim3(64), dim3(64), 0, 0, l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  hipLaunchKernelGGL(apply_grad, dim3(64), dim3(64), 0, 0, l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);

  end = clock();
  return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void learn(int iter)
{
  float err;

  double time_taken = 0.0;

  fprintf(stdout ,"Learning\n");

  while (iter < 0 || iter-- > 0) {
    err = 0.0f;

    for (unsigned int i = 0; i < train_cnt; ++i) {
      float tmp_err;

      time_taken += forward_pass(train_set[i].data);

      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      // Euclid distance of train_set[i]
      hipLaunchKernelGGL(makeError, dim3(10), dim3(1), 0, 0, l_f.d_preact, l_f.output, train_set[i].label, 10);
      snrm2(10, l_f.d_preact, tmp_err);
      err += tmp_err;

      time_taken += back_pass();
    }

    err /= train_cnt;
    fprintf(stdout, "error: %e, time_on_gpu: %lf\n", err, time_taken);

    if (err < threshold) {
      fprintf(stdout, "Training complete, error less than threshold\n\n");
      break;
    }

  }

  fprintf(stdout, "\n Time - %lf\n", time_taken);
}


// Returns label of given data (0-9)
static unsigned int classify(double data[28][28])
{
  float res[10];

  forward_pass(data);

  unsigned int max = 0;

  hipMemcpy(res, l_f.output, sizeof(float) * 10, hipMemcpyDeviceToHost);

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test()
{
  int error = 0;

  for (unsigned int i = 0; i < test_cnt; ++i) {
    if (classify(test_set[i].data) != test_set[i].label) {
      ++error;
    }
  }

  fprintf(stdout, "Error Rate: %.2lf%%\n",
      double(error) / double(test_cnt) * 100.0);
}
