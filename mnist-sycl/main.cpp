#include <cstdio>
#include <chrono>
#include "common.h"

#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "kernels.h"

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static inline void loaddata()
{
  mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
      &train_set, &train_cnt);
  mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
      &test_set, &test_cnt);
}

// replace cublas function in the case n = 10
void snrm2(queue &q, const int n, buffer<float, 1> &x, float &result) {
  if (n <= 0) {
    result = 0.f;
    return;
  }
  float *r = (float*) malloc (n * sizeof(float));

  q.submit([&] (handler &cgh) {
    auto acc = x.get_access<sycl_read>(cgh);
    cgh.copy(acc, r);
  }).wait();

  float sum = 0.f;
  for (int i = 0; i < n; i++) sum += r[i] * r[i];
  result = sqrtf(sum);
  free(r);
}

// Forward propagation of a single row in dataset
void forward_pass(
  queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  float input[28][28];
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      input[i][j] = data[i][j];
    }
  }

  l_input.clear(q);
  l_c1.clear(q);
  l_s1.clear(q);
  l_f.clear(q);

  l_input.setOutput(q, (float *)input);

  range<1> gws (64 * 64);
  range<1> lws (64);

  // fp_preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
  auto in_o_re = l_input.output.reinterpret<float[28]>(range<1>(28));
  auto c1_p_re = l_c1.preact.reinterpret<float[24][24]>(range<1>(6));
  auto c1_w_re = l_c1.weight.reinterpret<float[5][5]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto o = in_o_re.get_access<sycl_read>(cgh);
    auto p = c1_p_re.get_access<sycl_read_write>(cgh);
    auto w = c1_w_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class fw_preact_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_preact_c1(item, o.get_pointer(), p.get_pointer(), w.get_pointer());
    });
  });

  // fp_bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
  // auto c1_p_re = l_c1.preact.reinterpret<float[24][24]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto p = c1_p_re.get_access<sycl_read_write>(cgh);
    auto b = l_c1.bias.get_access<sycl_read>(cgh);
    cgh.parallel_for<class fw_bias_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_bias_c1(item, p.get_pointer(), b.get_pointer());
    });
  });

  // apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);
  const int l_c1_O = l_c1.O;
  q.submit([&] (handler &cgh) {
    auto p = l_c1.preact.get_access<sycl_read>(cgh);
    auto o = l_c1.output.get_access<sycl_write>(cgh);
    cgh.parallel_for<class c1_step>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_step_function(item, p.get_pointer(), o.get_pointer(), l_c1_O);
    });
  });

  // fp_preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
  auto c1_o_re = l_c1.output.reinterpret<float[24][24]>(range<1>(6));
  auto s1_p_re = l_s1.preact.reinterpret<float[6][6]>(range<1>(6));
  auto s1_w_re = l_s1.weight.reinterpret<float[4][4]>(range<1>(1));
  q.submit([&] (handler &cgh) {
    auto o = c1_o_re.get_access<sycl_read>(cgh);
    auto p = s1_p_re.get_access<sycl_read_write>(cgh);
    auto w = s1_w_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class preact_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_preact_s1(item, o.get_pointer(), p.get_pointer(), w.get_pointer());
    });
  });

  // fp_bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
  //auto s1_p_re = l_s1.preact.reinterpret<float[6][6]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto p = s1_p_re.get_access<sycl_read_write>(cgh);
    auto b = l_s1.bias.get_access<sycl_read>(cgh);
    cgh.parallel_for<class fw_bias_s1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_bias_s1(item, p.get_pointer(), b.get_pointer());
    });
  });


  // apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);
  const int l_s1_O = l_s1.O;
  q.submit([&] (handler &cgh) {
    auto p = l_s1.preact.get_access<sycl_read>(cgh);
    auto o = l_s1.output.get_access<sycl_write>(cgh);
    cgh.parallel_for<class s1_step>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_step_function(item, p.get_pointer(), o.get_pointer(), l_s1_O);
    });
  });

  //fp_preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
  auto s1_o_re = l_s1.output.reinterpret<float[6][6]>(range<1>(6));
  auto f_w_re = l_f.weight.reinterpret<float[6][6][6]>(range<1>(10));
  q.submit([&] (handler &cgh) {
    auto o = s1_o_re.get_access<sycl_read>(cgh);
    auto p = l_f.preact.get_access<sycl_read_write>(cgh);
    auto w = f_w_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class preact_f>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_preact_f(item, o.get_pointer(), p.get_pointer(), w.get_pointer());
    });
  });

  // fp_bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
  q.submit([&] (handler &cgh) {
    auto p = l_f.preact.get_access<sycl_read_write>(cgh);
    auto b = l_f.bias.get_access<sycl_read>(cgh);
    cgh.parallel_for<class fw_bias_f>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      fp_bias_f(item, p.get_pointer(), b.get_pointer());
    });
  });

  // apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
  const int l_f_O = l_f.O;
  q.submit([&] (handler &cgh) {
    auto p = l_f.preact.get_access<sycl_read>(cgh);
    auto o = l_f.output.get_access<sycl_write>(cgh);
    cgh.parallel_for<class f_step>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_step_function(item, p.get_pointer(), o.get_pointer(), l_f_O);
    });
  });
}

// Back propagation to update weights
void back_pass(
  queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
  range<1> gws (64 * 64);
  range<1> lws (64);

  // bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
  auto f_dw_re = l_f.d_weight.reinterpret<float[6][6][6]>(range<1>(10));
  auto s1_o_re = l_s1.output.reinterpret<float[6][6]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto dw = f_dw_re.get_access<sycl_write>(cgh);
    auto dp = l_f.d_preact.get_access<sycl_read>(cgh);
    auto o = s1_o_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_weight_f>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_weight_f(item, dw.get_pointer(), dp.get_pointer(), o.get_pointer());
    });
  });

  // bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);
  q.submit([&] (handler &cgh) {
    auto b = l_f.bias.get_access<sycl_read_write>(cgh);
    auto dp = l_f.d_preact.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_bias_f>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_bias_f(item, b.get_pointer(), dp.get_pointer());
    });
  });
    
  // bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
  auto s1_do_re = l_s1.d_output.reinterpret<float[6][6]>(range<1>(6));
  auto f_w_re = l_f.weight.reinterpret<float[6][6][6]>(range<1>(10));
  q.submit([&] (handler &cgh) {
    auto  o = s1_do_re.get_access<sycl_read_write>(cgh);
    auto  w = f_w_re.get_access<sycl_read>(cgh);
    auto dp = l_f.d_preact.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_output_s1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_output_s1(item, o.get_pointer(), w.get_pointer(), dp.get_pointer());
    });
  });

  //bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
  auto s1_dp_re = l_s1.d_preact.reinterpret<float[6][6]>(range<1>(6));
  //auto s1_do_re = l_s1.d_output.reinterpret<float[6][6]>(range<1>(6));
  auto s1_p_re = l_s1.preact.reinterpret<float[6][6]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto dp = s1_dp_re.get_access<sycl_write>(cgh);
    auto  o = s1_do_re.get_access<sycl_read>(cgh);
    auto  p = s1_p_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_preact_s1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_preact_s1(item, dp.get_pointer(), o.get_pointer(), p.get_pointer());
    });
  });

  // bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
  auto s1_dw_re = l_s1.d_weight.reinterpret<float[4][4]>(range<1>(1));
  //auto s1_dp_re = l_s1.d_preact.reinterpret<float[6][6]>(range<1>(6));
  auto c1_o_re = l_c1.output.reinterpret<float[24][24]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto dw = s1_dw_re.get_access<sycl_read_write>(cgh);
    auto dp = s1_dp_re.get_access<sycl_read>(cgh);
    auto o = c1_o_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_weight_s1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_weight_s1(item, dw.get_pointer(), dp.get_pointer(), o.get_pointer());
    });
  });

  // bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
  q.submit([&] (handler &cgh) {
    auto b = l_s1.bias.get_access<sycl_read_write>(cgh);
    auto dp = s1_dp_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_bias_s1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_bias_s1(item, b.get_pointer(), dp.get_pointer());
    });
  });

  // bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
  auto c1_do_re = l_c1.d_output.reinterpret<float[24][24]>(range<1>(6));
  auto s1_w_re = l_s1.weight.reinterpret<float[4][4]>(range<1>(1));
  //auto s1_dp_re = l_s1.d_preact.reinterpret<float[6][6]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto  o = c1_do_re.get_access<sycl_read_write>(cgh);
    auto  w = s1_w_re.get_access<sycl_read>(cgh);
    auto dp = s1_dp_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_output_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_output_c1(item, o.get_pointer(), w.get_pointer(), dp.get_pointer());
    });
  });

  // bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
  auto c1_dp_re = l_c1.d_preact.reinterpret<float[24][24]>(range<1>(6));
  //auto c1_do_re = l_c1.d_output.reinterpret<float[24][24]>(range<1>(6));
  auto c1_p_re = l_c1.preact.reinterpret<float[24][24]>(range<1>(6));
  q.submit([&] (handler &cgh) {
    auto dp = c1_dp_re.get_access<sycl_write>(cgh);
    auto  o = c1_do_re.get_access<sycl_read>(cgh);
    auto p = c1_p_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_preact_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_preact_c1(item, dp.get_pointer(), o.get_pointer(), p.get_pointer());
    });
  });

  // bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
  auto c1_dw_re = l_c1.d_weight.reinterpret<float[5][5]>(range<1>(6));
  // auto c1_dp_re = l_c1.d_preact.reinterpret<float[24][24]>(range<1>(6));
  auto i_o_re = l_input.output.reinterpret<float[28]>(range<1>(28));
  q.submit([&] (handler &cgh) {
    auto dw = c1_dw_re.get_access<sycl_read_write>(cgh);
    auto dp = c1_dp_re.get_access<sycl_read>(cgh);
    auto o = i_o_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_weight_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_weight_c1(item, dw.get_pointer(), dp.get_pointer(), o.get_pointer());
    });
  });
  
  // bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);
  q.submit([&] (handler &cgh) {
    auto b = l_c1.bias.get_access<sycl_read_write>(cgh);
    auto dp = c1_dp_re.get_access<sycl_read>(cgh);
    cgh.parallel_for<class bw_bias_c1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      bp_bias_c1(item, b.get_pointer(), dp.get_pointer());
    });
  });

  // apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
  const int l_f_mn = l_f.M * l_f.N;
  q.submit([&] (handler &cgh) {
    auto w = l_f.weight.get_access<sycl_read_write>(cgh);
    auto dw = l_f.d_weight.get_access<sycl_read>(cgh);
    cgh.parallel_for<class l_f_grad>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_grad(item, w.get_pointer(), dw.get_pointer(), l_f_mn); 
    });
  });

  const int l_s1_mn = l_s1.M * l_s1.N;
  // apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
  q.submit([&] (handler &cgh) {
    auto w = l_s1.weight.get_access<sycl_read_write>(cgh);
    auto dw = l_s1.d_weight.get_access<sycl_read>(cgh);
    cgh.parallel_for<class l_s1_grad>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_grad(item, w.get_pointer(), dw.get_pointer(), l_s1_mn); 
    });
  });

  const int l_c1_mn = l_c1.M * l_c1.N;
  // apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
  q.submit([&] (handler &cgh) {
    auto w = l_c1.weight.get_access<sycl_read_write>(cgh);
    auto dw = l_c1.d_weight.get_access<sycl_read>(cgh);
    cgh.parallel_for<class l_c1_grad>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      apply_grad(item, w.get_pointer(), dw.get_pointer(), l_c1_mn); 
    });
  });
}

static void learn(
  queue &q,
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

      l_f.bp_clear(q);
      l_s1.bp_clear(q);
      l_c1.bp_clear(q);

      // Euclid distance of train_set[i]
      range<1> gws (10);
      range<1> lws (1);
      // makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);
      const unsigned int train_set_label = train_set[i].label;
      q.submit([&] (handler &cgh) {
        auto dp = l_f.d_preact.get_access<sycl_write>(cgh);
        auto o = l_f.output.get_access<sycl_read>(cgh);
        cgh.parallel_for<class err>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          makeError(item, dp.get_pointer(), o.get_pointer(), train_set_label, 10);
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
  queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f,
  double data[28][28])
{
  float res[10];

  forward_pass(q, l_input, l_c1, l_s1, l_f, data);

  unsigned int max = 0;

  q.submit([&] (handler &cgh) {
    auto acc = l_f.output.get_access<sycl_read>(cgh);
    cgh.copy(acc, res);
  }).wait();

  for (int i = 1; i < 10; ++i) {
    if (res[max] < res[i]) {
      max = i;
    }
  }

  return max;
}

// Perform forward propagation of test data
static void test(
  queue &q,
  Layer &l_input,
  Layer &l_c1,
  Layer &l_s1,
  Layer &l_f)
{
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
  loaddata();

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

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
