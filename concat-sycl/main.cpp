#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

const int max_seq_len = 1024;
const int max_batch_tokens = 9216;

/* Convert 3-dim tensor index into vector index */
inline int flat_3dim(int id1, int id2, int id3, int dim2, int dim3) {
  return id1 * dim2 * dim3 + id2 * dim3 + id3;
}

template <typename T>
void concat (sycl::nd_item<1> &item,
             const T *__restrict inp1,
             const T *__restrict inp2,
                   T *output,
             int sz0, int sz2, int sz1_1, int sz1_2)
{
  int nele = sz0 * sz2 * (sz1_1 + sz1_2);
  int idx = item.get_global_id(0);
  if (idx >= nele) return;

  float *dst_ptr = (float *)output + idx;
  int idx2 = idx % sz2;
  idx = idx / sz2;
  int idx1 = idx % (sz1_1 + sz1_2);
  int idx0 = idx / (sz1_1 + sz1_2);
  float *src_ptr = nullptr;
  int sz1 = 0;
  if (idx1 < sz1_1) {
    sz1 = sz1_1;
    src_ptr = (float *)inp1;
  } else {
    idx1 -= sz1_1;
    sz1 = sz1_2;
    src_ptr = (float *)inp2;
  }
  src_ptr += flat_3dim(idx0, idx1, idx2, sz1, sz2);
  *dst_ptr = *src_ptr;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int nhead = 4; nhead <= 16; nhead += 4) { // a multiple of 4
    srand(nhead);

    int seq_len = rand() % max_seq_len + 1;
    while (seq_len <= 1) {
      seq_len = rand() % max_seq_len + 1;
    }

    const int max_batch_size = max_batch_tokens / seq_len;
    const int batch_size = rand() % max_batch_size + 1;
    const int upbound = 1024 / nhead + 1;
    const int hidden_dim = (rand() % upbound + 1) * nhead * 4;
    const int head_dim = hidden_dim / nhead;

    const int sl1 = rand() % (seq_len - 1) + 1;
    const int sl2 = seq_len - sl1;
    const int beam_size = rand() % 8 + 1;

    printf("num_head = %d\t", nhead);
    printf("seq_len = %d\t", seq_len);
    printf("batch size = %d\t", batch_size);
    printf("hidden dimension = %d\t", hidden_dim);
    printf("beam size = %d\n", beam_size);

    const size_t inp1_size = batch_size * beam_size * nhead * sl1 * head_dim;
    const size_t inp2_size = batch_size * beam_size * nhead * sl2 * head_dim;
    const size_t outp_size = batch_size * beam_size * nhead * seq_len * head_dim;

    const size_t inp1_size_bytes = inp1_size * sizeof(float);
    const size_t inp2_size_bytes = inp2_size * sizeof(float);
    const size_t outp_size_bytes = outp_size * sizeof(float);

    float *inp1 = (float*) malloc (inp1_size_bytes);
    float *inp2 = (float*) malloc (inp2_size_bytes);
    float *outp = (float*) malloc (outp_size_bytes);

    for (size_t i = 0; i < inp1_size; i++) {
      inp1[i] = -1.f;
    }

    for (size_t i = 0; i < inp2_size; i++) {
      inp2[i] = 1.f;
    }

    float *d_inp1, *d_inp2, *d_outp;

    d_inp1 = (float *)sycl::malloc_device(inp1_size_bytes, q);
    d_inp2 = (float *)sycl::malloc_device(inp2_size_bytes, q);
    d_outp = (float *)sycl::malloc_device(outp_size_bytes, q);
    q.memcpy(d_inp1, inp1, inp1_size_bytes);
    q.memcpy(d_inp2, inp2, inp2_size_bytes);

    const size_t n = batch_size * beam_size * nhead * head_dim * (sl1 + sl2);
    const size_t nblock = (n + 255) / 256;
    sycl::range<1> gws (nblock * 256);
    sycl::range<1> lws (256);

    // warmup
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class warmup>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        concat(item, d_inp1, d_inp2, d_outp, batch_size * beam_size * nhead, head_dim, sl1, sl2);
      });
    });

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class eval>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          concat(item, d_inp1, d_inp2, d_outp, batch_size * beam_size * nhead, head_dim, sl1, sl2);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(outp, d_outp, outp_size_bytes).wait();

    double checksum = 0;
    for (size_t i = 0; i < outp_size; i++) {
      checksum += outp[i];
    }
    printf("Checksum = %lf\n\n", checksum);

    sycl::free(d_inp1, q);
    sycl::free(d_inp2, q);
    sycl::free(d_outp, q);
    free(inp1);
    free(inp2);
    free(outp);
  }
}
