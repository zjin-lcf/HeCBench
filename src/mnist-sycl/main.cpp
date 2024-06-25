#include <cstdio>
#include <chrono>
#include <sycl/sycl.hpp>

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "kernels.h"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static inline int loaddata()
{
  int s1 = mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
                      &train_set, &train_cnt);
  int s2 = mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
                      &test_set, &test_cnt);
  return s1 | s2;
}

// replace cublas function in the case n = 10
void snrm2(sycl::queue &q, const int n, float *x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));
  q.memcpy(r, x, n * sizeof(float)).wait();

  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

// Forward propagation of a single row in dataset
void forward_pass(
  sycl::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  l_input.clear();
  l_c1.clear();
  l_s1.clear();
  l_f.clear();

  float input[28][28];
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }
  l_input.setOutput((float *)input);

  sycl::range<1> gws (64 * 64);
  sycl::range<1> lws (64);

  // fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  q.submit([&] (sycl::handler &cgh) {
    auto o = (float (*)[28])l_input.output;
    auto p = (float (*)[24][24])l_c1.preact;
    auto w = (float (*)[5][5])l_c1.weight;
    cgh.parallel_for<class fw_preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_c1(item, o, p, w);
    });
  });

  // fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
  // auto c1_p_re = l_c1.preact.reinterpret<float[24][24]>(range<1>(6));
  q.submit([&] (sycl::handler &cgh) {
    auto p = (float (*)[24][24])l_c1.preact;
    auto b = l_c1.bias;
    cgh.parallel_for<class fw_bias_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_c1(item, p, b);
    });
  });

  // apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);
  q.submit([&] (sycl::handler &cgh) {
    auto p = l_c1.preact;
    auto o = l_c1.output;
    auto l_c1_O = l_c1.O;
    cgh.parallel_for<class c1_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_c1_O);
    });
  });

  // fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  q.submit([&] (sycl::handler &cgh) {
    auto o = (float (*)[24][24])l_c1.output;
    auto p = (float (*)[6][6])l_s1.preact;
    auto w = (float (*)[4][4])l_s1.weight;
    cgh.parallel_for<class preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_s1(item, o, p, w);
    });
  });

  // fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
  //auto s1_p_re = l_s1.preact.reinterpret<float[6][6]>(range<1>(6));
  q.submit([&] (sycl::handler &cgh) {
    auto p = (float (*)[6][6])l_s1.preact;
    auto b = l_s1.bias;
    cgh.parallel_for<class fw_bias_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_s1(item, p, b);
    });
  });


  // apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);
  const int l_s1_O = l_s1.O;
  q.submit([&] (sycl::handler &cgh) {
    auto p = l_s1.preact;
    auto o = l_s1.output;
    cgh.parallel_for<class s1_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_s1_O);
    });
  });

  //fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  q.submit([&] (sycl::handler &cgh) {
    auto o = (float (*)[6][6])l_s1.output;
    auto p = l_f.preact;
    auto w = (float (*)[6][6][6])l_f.weight;
    cgh.parallel_for<class preact_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_preact_f(item, o, p, w);
    });
  });

  // fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  q.submit([&] (sycl::handler &cgh) {
    auto p = l_f.preact;
    auto b = l_f.bias;
    cgh.parallel_for<class fw_bias_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      fp_bias_f(item, p, b);
    });
  });

  // apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
  q.submit([&] (sycl::handler &cgh) {
    auto p = l_f.preact;
    auto o = l_f.output;
    auto l_f_O = l_f.O;
    cgh.parallel_for<class f_step>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_step_function(item, p, o, l_f_O);
    });
  });
}

// Back propagation to update weights
void back_pass(
  sycl::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
  sycl::range<1> gws (64 * 64);
  sycl::range<1> lws (64);

  // bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  q.submit([&] (sycl::handler &cgh) {
    auto dw = (float (*)[6][6][6])l_f.d_weight;
    auto dp = l_f.d_preact;
    auto o = (float (*)[6][6])l_s1.output;
    cgh.parallel_for<class bw_weight_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_f(item, dw, dp, o);
    });
  });

  // bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);
  q.submit([&] (sycl::handler &cgh) {
    auto b = l_f.bias;
    auto dp = l_f.d_preact;
    cgh.parallel_for<class bw_bias_f>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_f(item, b, dp);
    });
  });

  // bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  q.submit([&] (sycl::handler &cgh) {
    auto  o = (float (*)[6][6])l_s1.d_output;
    auto  w = (float (*)[6][6][6])l_f.weight;
    auto dp = l_f.d_preact;
    cgh.parallel_for<class bw_output_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_output_s1(item, o, w, dp);
    });
  });

  //bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  q.submit([&] (sycl::handler &cgh) {
    auto dp = (float (*)[6][6])l_s1.d_preact;
    auto  o = (float (*)[6][6])l_s1.d_output;
    auto  p = (float (*)[6][6])l_s1.preact;
    cgh.parallel_for<class bw_preact_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_preact_s1(item, dp, o, p);
    });
  });

  // bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  q.submit([&] (sycl::handler &cgh) {
    auto dw = (float (*)[4][4])l_s1.d_weight;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    auto o = (float (*)[24][24])l_c1.output;
    cgh.parallel_for<class bw_weight_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_s1(item, dw, dp, o);
    });
  });

  // bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
  q.submit([&] (sycl::handler &cgh) {
    auto b = l_s1.bias;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    cgh.parallel_for<class bw_bias_s1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_s1(item, b, dp);
    });
  });

  // bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  q.submit([&] (sycl::handler &cgh) {
    auto  o = (float (*)[24][24])l_c1.d_output;
    auto  w = (float (*)[4][4])l_s1.weight;
    auto dp = (float (*)[6][6])l_s1.d_preact;
    cgh.parallel_for<class bw_output_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_output_c1(item, o, w, dp);
    });
  });

  // bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  q.submit([&] (sycl::handler &cgh) {
    auto dp = (float (*)[24][24])l_c1.d_preact;
    auto  o = (float (*)[24][24])l_c1.d_output;
    auto p = (float (*)[24][24])l_c1.preact;
    cgh.parallel_for<class bw_preact_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_preact_c1(item, dp, o, p);
    });
  });

  // bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  q.submit([&] (sycl::handler &cgh) {
    auto dw = (float (*)[5][5])l_c1.d_weight;
    auto dp = (float (*)[24][24])l_c1.d_preact;
    auto o = (float (*)[28])l_input.output;
    cgh.parallel_for<class bw_weight_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_weight_c1(item, dw, dp, o);
    });
  });

  // bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
  q.submit([&] (sycl::handler &cgh) {
    auto b = l_c1.bias;
    auto dp = (float (*)[24][24])l_c1.d_preact;
    cgh.parallel_for<class bw_bias_c1>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bp_bias_c1(item, b, dp);
    });
  });

  // apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  q.submit([&] (sycl::handler &cgh) {
    auto w = l_f.weight;
    auto dw = l_f.d_weight;
    auto l_f_mn = l_f.M * l_f.N;
    cgh.parallel_for<class l_f_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_f_mn);
    });
  });

  // apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  q.submit([&] (sycl::handler &cgh) {
    auto w = l_s1.weight;
    auto dw = l_s1.d_weight;
    auto l_s1_mn = l_s1.M * l_s1.N;
    cgh.parallel_for<class l_s1_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_s1_mn);
    });
  });

  // apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
  q.submit([&] (sycl::handler &cgh) {
    auto w = l_c1.weight;
    auto dw = l_c1.d_weight;
    auto l_c1_mn = l_c1.M * l_c1.N;
    cgh.parallel_for<class l_c1_grad>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      apply_grad(item, w, dw, l_c1_mn);
    });
  });
}

static void learn(
  sycl::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  int iter)
{
  float err;
  fprintf(stdout ,"Learning\n");

  while (iter < 0 || iter-- > 0) {
    err = 0.0f;

    for (unsigned int i = 0; i < train_cnt; ++i) {
      float tmp_err;

      forward_pass(q, l_input, l_c1, l_s1, l_f, train_set[i].data);

      l_f.bp_clear();
      l_s1.bp_clear();
      l_c1.bp_clear();

      // Euclid distance of train_set[i]
      sycl::range<1> gws (10);
      sycl::range<1> lws (1);
      // makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
      q.submit([&] (sycl::handler &cgh) {
        auto dp = l_f.d_preact;
        auto o = l_f.output;
        auto train_set_label = train_set[i].label;
        cgh.parallel_for<class err>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          makeError(item, dp, o, train_set_label, 10);
        });
      });

      snrm2(q, 10, l_f.d_preact, tmp_err);
      err += tmp_err;

      back_pass(q, l_input, l_c1, l_s1, l_f);
    }

    err /= train_cnt;
    fprintf(stdout, "error: %e\n", err);

    if (err < threshold) {
      fprintf(stdout, "Training complete, error less than threshold\n\n");
      break;
    }
  }
}


// Returns label of given data (0-9)
static unsigned int classify(
  sycl::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  float res[10];

  forward_pass(q, l_input, l_c1, l_s1, l_f, data);

  unsigned int max = 0;

  q.memcpy(res, l_f.output, sizeof(float) * 10).wait();

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test(
  sycl::queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
  fprintf(stdout ,"Testing\n");

  int error = 0;

  for (unsigned int i = 0; i < test_cnt; ++i) {
    if (classify(q, l_input, l_c1, l_s1, l_f, test_set[i].data)
        != test_set[i].label) {
      ++error;
    }
  }

  fprintf(stdout, "Error Rate: %.2lf%%\n",
      double(error) / double(test_cnt) * 100.0);
}

int main(int argc, const  char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <iterations>\n", argv[0]);
    return 1;
  }

  const int iter = atoi(argv[1]);
  srand(123);
  if (loaddata() != 0) return 1;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Layer l_input (q, 0, 0, 28*28);
  Layer l_c1 (q, 5*5, 6, 24*24*6);
  Layer l_s1 (q, 4*4, 1, 6*6*6);
  Layer l_f (q, 6*6*6, 10, 10);

  auto t1 = std::chrono::high_resolution_clock::now();
  learn(q, l_input, l_c1, l_s1, l_f, iter);
  test(q, l_input, l_c1, l_s1, l_f);
  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Total time (learn + test) %lf secs \n", total_time / 1.0e6);
  return 0;
}
