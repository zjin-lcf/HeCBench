#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <omp.h>
#include "reference.h"
#include "common.h"

// GPU thread block size
#define TPB 1024

template<class Kernel, class... KernelArgs>
float benchmark_kernel(int repeat, Kernel kernel, KernelArgs&&... kernel_args) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    kernel(std::forward<KernelArgs>(kernel_args)...);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return (time * 1e-3f) / repeat;
}

template<class D, class T>
void validate_result(D* out_gpu, const T* cpu_reference,
                     const char* name, std::uint64_t num_elements,
                     T tolerance=1e-4, int n_print=5, int check_all=0)
{
  #pragma omp target update from (out_gpu[0:num_elements])
  int nfaults = 0;
  for (uint64_t i = 0; i < num_elements; i++) {
    if (std::fabs(cpu_reference[i] - (T)out_gpu[i]) > tolerance && std::isfinite(cpu_reference[i])) {
      printf("Mismatch of %s at %zu: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], (T)out_gpu[i]);
      nfaults++;
      if (nfaults >= max_int(10, n_print)) {
        break;
      }
    }
  }
  printf("%s\n", (nfaults == 0) ? "PASS" : "FAIL");
}

// -----------------------------------------------------------------------------------------------
// GPU kernels

// using kernel 5 because for images, each "channel" is effectively
// H * W * group_size, which is quite large
// One block per group of group_size channels: B * C / group_size (B * n_groups) blocks
void groupnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int B, int C, int img_size, int group_size, int n_groups,
    int team_size,
    int block_size
) {
    const int group_pixels = img_size * group_size;   // pixels per group per image
    const float eps = 1e-5f;

    #pragma omp target teams distribute collapse(2) num_teams(team_size)
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < n_groups; g++) {
            int block_idx = b * n_groups + g;

            const float* x_block      = x   + block_idx * group_pixels;
                  float* out_block     = out  + block_idx * group_pixels;
            const float* weight_group  = weight + g * group_size;
            const float* bias_group    = bias   + g * group_size;

            float sum  = 0.0f;
            float sum2 = 0.0f;
            #pragma omp parallel for reduction(+:sum, sum2) num_threads(block_size)
            for (int i = 0; i < group_pixels; i++) {
                float val = x_block[i];
                sum  += val;
                sum2 += val * val;
            }

            float m   = sum  / group_pixels;
            float m2  = sum2 / group_pixels;
            float var = m2 - m * m;                // E[x²] - E[x]²
            float s   = 1.0f / sqrtf(var + eps);  // rsqrtf equivalent

            if (mean != nullptr) mean[block_idx] = m;
            if (rstd != nullptr) rstd[block_idx] = s;

            #pragma omp parallel for num_threads(block_size)
            for (int i = 0; i < group_pixels; i++) {
                int c = i / img_size;
                float n   = s * (x_block[i] - m);
                out_block[i] = n * weight_group[c] + bias_group[c];
            }
        }
    }
}

void groupnorm_backward_kernel(
    const float* __restrict__ dout,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ weight,
    float* __restrict__ dx,
    float* __restrict__ dweight,
    float* __restrict__ dbias,
    int B, int C, int img_size, int group_size, int n_groups,
    int team_size,
    int block_size
) {
    const int group_pixels = img_size * group_size;

    #pragma omp target teams distribute collapse(2) num_teams(team_size)
    for (int b = 0; b < B; b++) {
        for (int g = 0; g < n_groups; g++) {
            int block_idx = b * n_groups + g;

            const float* dout_block  = dout   + block_idx * group_pixels;
            const float* x_block     = x      + block_idx * group_pixels;
                  float* dx_block    = dx     + block_idx * group_pixels;
            const float* weight_g    = weight  + g * group_size;
                  float* dweight_g   = dweight + g * group_size;
                  float* dbias_g     = dbias   + g * group_size;

            float m_val    = mean[block_idx];
            float rstd_val = rstd[block_idx];

            float w_dout_sum      = 0.0f;
            float w_dout_norm_sum = 0.0f;

            #pragma omp parallel for reduction(+:w_dout_sum, w_dout_norm_sum) \
             num_threads(block_size)
            for (int i = 0; i < group_pixels; i++) {
                int   c = i / img_size;
                float cur_w_dout   = weight_g[c] * dout_block[i];
                w_dout_sum        += cur_w_dout;
                float norm         = (x_block[i] - m_val) * rstd_val;
                w_dout_norm_sum   += cur_w_dout * norm;
            }

            float w_dout_block      = w_dout_sum      / group_pixels;
            float w_dout_norm_block = w_dout_norm_sum / group_pixels;

            // compute dx
            #pragma omp parallel for num_threads(block_size)
            for (int i = 0; i < group_pixels; i++) {
                int   c = i / img_size;
                float dout_val    = dout_block[i];
                float norm        = (x_block[i] - m_val) * rstd_val;
                float w_dout      = weight_g[c] * dout_val;
                dx_block[i]       = (w_dout - w_dout_block - norm * w_dout_norm_block) * rstd_val;
            }

            // compute dweight and dbias
            for (int c = 0; c < group_size; c++) {
                const float* dout_ch = dout_block + c * img_size;
                const float* x_ch   = x_block    + c * img_size;

                float dw = 0.0f;
                float db = 0.0f;
                #pragma omp parallel for reduction(+:dw, db) num_threads(block_size)
                for (int i = 0; i < img_size; i++) {
                    float dout_val  = dout_ch[i];
                    db             += dout_val;
                    float norm      = (x_ch[i] - m_val) * rstd_val;
                    dw             += dout_val * norm;
                }
                #pragma omp atomic update
                dweight_g[c] += dw;
                #pragma omp atomic update
                dbias_g[c]   += db;
            }
        }
    }
}

// kernel launcher

void groupnorm_forward(
    const float* x, const float* weight, const float* bias,
    float* out, float* mean, float* rstd,
    int B, int C, int img_size, int n_groups
) {
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32);
    groupnorm_forward_kernel(
        x, weight, bias, out, mean, rstd, B, C, img_size, group_size, n_groups,
        n_blocks, block_size
    );
}

void groupnorm_backward(
    const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C, int img_size, int n_groups
) {
    int group_size = C / n_groups;
    int n_blocks = B * n_groups;
    int block_size = max_int(min_int(TPB, img_size * group_size), 32 * group_size);
    groupnorm_backward_kernel(
        dout, x, mean, rstd, weight, dx, dweight, dbias, B, C, img_size, group_size, n_groups,
        n_blocks, block_size
    );
}


// -----------------------------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 7) {
      printf("Usage: %s <batch size> <number of channels> <height> <width> <number of groups> <repeat>\n", argv[0]);
      return 1;
    }

    uint64_t B = atoi(argv[1]);
    uint64_t C = atoi(argv[2]);
    uint64_t H = atoi(argv[3]);
    uint64_t W = atoi(argv[4]);
    uint64_t n_groups = atoi(argv[5]);
    int repeat = atoi(argv[6]);

    uint64_t img_size = H * W;

    srand(0);
    float *d_x = make_random_float(B * C * img_size);
    float *d_weight = make_random_float(C);
    float *d_bias = make_random_float(C);
    float *d_dout = make_random_float(B * C * img_size);

    // host results
    float *dx = (float*)calloc(B * C * img_size , sizeof(float));
    float *dweight = (float*)calloc(C , sizeof(float));
    float *dbias = (float*)calloc(C , sizeof(float));

    float *out = (float*)malloc(B * C * img_size * sizeof(float));
    float *mean = (float*) malloc(B * n_groups * sizeof(float));
    float *rstd = (float*) malloc(B * n_groups * sizeof(float));

    // device results
    float *d_dx = (float*)calloc(B * C * img_size , sizeof(float));
    float *d_dweight = (float*)calloc(C , sizeof(float));
    float *d_dbias = (float*)calloc(C , sizeof(float));

    float *d_out = (float*)malloc(B * C * img_size * sizeof(float));
    float *d_mean = (float*) malloc(B * n_groups * sizeof(float));
    float *d_rstd = (float*) malloc(B * n_groups * sizeof(float));

    #pragma omp target data map(to: d_x[0:B*C*img_size], d_weight[0:C], \
                                    d_bias[0:C], d_dout[0:B*C*img_size], \
                                    d_dx[0:B*C*img_size], d_dweight[0:C], \
                                    d_dbias[0:C]) \
                            map(alloc: d_mean[0:B * n_groups], \
                                       d_rstd[0:B * n_groups], \
                                       d_out[0:B * C * img_size])
    {
      printf("Checking forward pass\n");

      groupnorm_forward_ref(d_x, d_weight, d_bias, out, mean, rstd,
                            B, C, img_size, n_groups);

      groupnorm_forward(d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                        B, C, img_size, n_groups);

      float fwd_acc = 1e-2;
      validate_result(d_out, out, "out", B * C * img_size, fwd_acc);

      printf("Checking backward pass\n");

      groupnorm_backward_ref(d_dout, d_x, mean, rstd, d_weight, dx, dweight, dbias,
                             B, C, img_size, n_groups);

      groupnorm_backward(d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                         B, C, img_size, n_groups);
      float acc = 1e-2;
      printf("Checking dbias\n");
      validate_result(d_dbias, dbias, "dbias", C, acc);
      printf("Checking dweight\n");
      validate_result(d_dweight, dweight, "dweight", C, acc);
      printf("Checking dx\n");
      validate_result(d_dx, dx, "dx", B * C * img_size, 1.0f);
      printf("\n─────────────────────────────────────────────────────\n");

      printf("Forward pass benchmarks\n");
      float elapsed_time = benchmark_kernel(repeat, groupnorm_forward,
                                            d_x, d_weight, d_bias, d_out, d_mean, d_rstd,
                                            B, C, img_size, n_groups);
      printf("time %.4f us\n", elapsed_time);

      printf("Backward pass benchmarks\n");
      elapsed_time = benchmark_kernel(repeat, groupnorm_backward,
                                      d_dout, d_x, d_mean, d_rstd, d_weight, d_dx, d_dweight, d_dbias,
                                      B, C, img_size, n_groups);
      printf("time %.4f us\n", elapsed_time);
    }

    free(d_x);
    free(d_weight);
    free(d_bias);
    free(d_out);
    free(d_dout);
    free(d_dx);
    free(d_dweight);
    free(d_dbias);
    free(d_mean);
    free(d_rstd);

    return 0;
}
