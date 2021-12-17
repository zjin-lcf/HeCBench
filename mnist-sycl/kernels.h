#include "layer.h"

// Constructor
Layer::Layer (queue &q, int M, int N, int O)
: output {O},
  preact {O},
  bias {N},
  weight {M*N},
  d_output {O},
  d_preact {O},
  d_weight {M*N} 
{
  this->M = M;
  this->N = N;
  this->O = O;

  float h_bias[N];
  float h_weight[N*M];

  for (int i = 0; i < N; ++i) {
    h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

    for (int j = 0; j < M; ++j)
      h_weight[i*M+j] = 0.5f - float(rand()) / float(RAND_MAX);
  }

  if (N != 0) {
    q.submit([&] (handler &cgh) {
      auto acc = bias.get_access<sycl_discard_write>(cgh);
      cgh.copy(h_bias, acc);
    });
  }

  if (N != 0 && M != 0) {
    q.submit([&] (handler &cgh) {
      auto acc = weight.get_access<sycl_discard_write>(cgh);
      cgh.copy(h_weight, acc);
    });
  }
}

// Send data one row from dataset to the GPU
void Layer::setOutput(queue &q, float *data)
{
  q.submit([&] (handler &cgh) {
    auto acc = output.get_access<sycl_discard_write>(cgh);
    cgh.copy(data, acc); 
  });
}

// Reset GPU memory between iterations
void Layer::clear(queue &q)
{
  q.submit([&] (handler &cgh) {
    auto acc = output.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });
  q.submit([&] (handler &cgh) {
    auto acc = preact.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });
}

void Layer::bp_clear(queue &q)
{
  q.submit([&] (handler &cgh) {
    auto acc = d_output.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });
  q.submit([&] (handler &cgh) {
    auto acc = d_preact.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });
  q.submit([&] (handler &cgh) {
    auto acc = d_weight.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0.f);
  });
}

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
inline void atomicFetchAdd(T& val, const T delta)
{
  sycl::ext::oneapi::atomic_ref<T, sycl::memory_order::relaxed, MemoryScope, 
                                sycl::access::address_space::global_space> 
    ref(val);
    ref.fetch_add(delta);
}

float step_function(float v)
{
  return 1.f / (1.f + sycl::exp(-v));
}

void apply_step_function(
  nd_item<1> &item,
  const float *__restrict input,
        float *__restrict output,
  const int N)
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    output[idx] = step_function(input[idx]);
  }
}

void makeError(
  nd_item<1> &item,
        float *__restrict err,
  const float *__restrict output,
  unsigned int Y, const int N)
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
  }
}

void apply_grad(
  nd_item<1> &item,
        float *__restrict output,
  const float *__restrict grad,
  const int N)
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);

  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    output[idx] += dt * grad[idx];
  }
}

void fp_preact_c1(
  nd_item<1> &item, 
  const float input[28][28],
        float preact[6][24][24],
  const float weight[6][5][5])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 5*5*6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1 ) % 5);
    const int i2 = ((idx /= 5 ) % 5);
    const int i3 = ((idx /= 5 ) % 6);
    const int i4 = ((idx /= 6 ) % 24);
    const int i5 = ((idx /= 24) % 24);
    atomicFetchAdd(preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
  }
}

void fp_bias_c1(nd_item<1> &item, float preact[6][24][24], const float bias[6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    preact[i1][i2][i3] += bias[i1];
  }
}

void fp_preact_s1(
  nd_item<1> &item,
  const float input[6][24][24],
        float preact[6][6][6],
  const float weight[1][4][4])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 4);
    const int i2 = ((idx /= 4  ) % 4);
    const int i3 = ((idx /= 4  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    atomicFetchAdd(preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
  }
}

void fp_bias_s1(nd_item<1> &item,float preact[6][6][6], const float bias[1])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    preact[i1][i2][i3] += bias[0];
  }
}

void fp_preact_f(
  nd_item<1> &item,
  const float input[6][6][6],
        float preact[10],
  const float weight[10][6][6][6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    atomicFetchAdd(preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
  }
}

void fp_bias_f(nd_item<1> &item, float preact[10], const float bias[10])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    preact[idx] += bias[idx];
  }
}

void bp_weight_f(
  nd_item<1> &item,
        float d_weight[10][6][6][6],
  const float d_preact[10],
  const float p_output[6][6][6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
  }
}

void bp_bias_f(nd_item<1> &item, float bias[10], const float d_preact[10])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 10;
  for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
    bias[idx] += dt * d_preact[idx];
  }
}

void bp_output_s1(
  nd_item<1> &item,
        float d_output[6][6][6],
  const float n_weight[10][6][6][6],
  const float nd_preact[10])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 10*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 10);
    const int i2 = ((idx /= 10  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const int i4 = ((idx /= 6  ) % 6);
    atomicFetchAdd(d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
  }
}

void bp_preact_s1(
  nd_item<1> &item,
        float d_preact[6][6][6],
  const float d_output[6][6][6],
  const float preact[6][6][6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    const float o = step_function(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

void bp_weight_s1(
  nd_item<1> &item,
        float d_weight[1][4][4],
  const float d_preact[6][6][6],
  const float p_output[6][24][24])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 1*4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 1);
    const int i2 = ((idx /= 1  ) % 4);
    const int i3 = ((idx /= 4  ) % 4);
    const int i4 = ((idx /= 4  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    const int i6 = ((idx /= 6  ) % 6);
    atomicFetchAdd(d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
  }
}

void bp_bias_s1(nd_item<1> &item, float bias[1], const float d_preact[6][6][6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*6*6;
  const float d = 216; //pow(6.0f, 3.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 6);
    const int i3 = ((idx /= 6  ) % 6);
    atomicFetchAdd(bias[0], dt * d_preact[i1][i2][i3] / d);
  }
}

void bp_output_c1(
  nd_item<1> &item,
        float d_output[6][24][24],
  const float n_weight[1][4][4],
  const float nd_preact[6][6][6])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 1*4*4*6*6*6;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 1);
    const int i2 = ((idx /= 1  ) % 4);
    const int i3 = ((idx /= 4  ) % 4);
    const int i4 = ((idx /= 4  ) % 6);
    const int i5 = ((idx /= 6  ) % 6);
    const int i6 = ((idx /= 6  ) % 6);
    atomicFetchAdd(d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
  }
}

void bp_preact_c1(
  nd_item<1> &item,
        float d_preact[6][24][24],
  const float d_output[6][24][24],
  const float preact[6][24][24])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*24*24;

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    const float o = step_function(preact[i1][i2][i3]);
    d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
  }
}

void bp_weight_c1(
  nd_item<1> &item,
        float d_weight[6][5][5],
  const float d_preact[6][24][24],
  const float p_output[28][28])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*5*5*24*24;
  const float d = 576; //pow(24.0f, 2.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 5);
    const int i3 = ((idx /= 5  ) % 5);
    const int i4 = ((idx /= 5  ) % 24);
    const int i5 = ((idx /= 24  ) % 24);
    atomicFetchAdd(d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
  }
}

void bp_bias_c1(nd_item<1> &item, float bias[6], const float d_preact[6][24][24])
{
  const int pos = item.get_global_id(0);
  const int size = item.get_local_range(0) * item.get_group_range(0);
  const int N = 6*24*24;
  const float d = 576; //pow(24.0f, 2.0f);

  for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
    int idx = n;
    const int i1 = ((idx /= 1  ) % 6);
    const int i2 = ((idx /= 6  ) % 24);
    const int i3 = ((idx /= 24  ) % 24);
    atomicFetchAdd(bias[i1], dt * d_preact[i1][i2][i3] / d);
  }
}
