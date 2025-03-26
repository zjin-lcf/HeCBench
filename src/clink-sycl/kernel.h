// a is a multiple of WGS for simplicity
#define N 8192
#define WGS 256
#define SAMPLE_TEST_LEN 20000

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

void lstm_inference(
  sycl::queue &q,
  sycl::range<3> &gws,
  sycl::range<3> &lws,
  const int slm_size,
  const float*__restrict__ d_x,
  const float*__restrict__ d_inW,
  const float*__restrict__ d_intW,
  const float*__restrict__ d_intB,
  const float*__restrict__ d_outW,
  const float*__restrict__ d_outB,
        float*__restrict__ d_y)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int t,i,j;
      int gid = item.get_global_id(2);

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
          i_state[j] = 1.f / (1.f + sycl::exp(-i_state[j]));
        }

        for (j = 0; j < 5; ++j) {
          f_state[j] = ldg(&d_inW[5+j]) * x;
          for (i = 0; i < 5; ++i)
            f_state[j] += h_state[i] * ldg(&d_intW[25+j*5+i]);
          f_state[j] += ldg(&d_intB[5+j]);
          f_state[j] = 1.f / (1.f + sycl::exp(-f_state[j]));
        }

        for (j = 0; j < 5; ++j) {
          o_state[j] = ldg(&d_inW[10+j]) * x;
          for (i = 0; i < 5; ++i)
            o_state[j] += h_state[i] * ldg(&d_intW[50+j*5+i]);
          o_state[j] += ldg(&d_intB[10+j]);
          o_state[j] = 1.f / (1.f + sycl::exp(-o_state[j]));
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
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
