#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

std::default_random_engine g;
// Results are inf when the range is [0,1]
std::uniform_real_distribution<float> distr (-1.f, 1.f);


// verification of FP32 MLP
using scalar_t = float;

// -----------------------------------------------------------------------------
// Column-major indexing helper
// -----------------------------------------------------------------------------
inline int idx_col_major(int row, int col, int ld)
{
  return row + col * ld;
}

// -----------------------------------------------------------------------------
// CPU GEMM equivalent of:
//
// C = W^T * X
//
// W: [ifeat x ofeat]
// X: [ifeat x batch]
// C: [ofeat x batch]
//
// All matrices are column-major.
// -----------------------------------------------------------------------------
void gemm_w_transpose(
    const scalar_t* W,
    const scalar_t* X,
    scalar_t* C,
    int ifeat,
    int ofeat,
    int batch_size)
{
  for (int col = 0; col < batch_size; col++) {
    for (int row = 0; row < ofeat; row++) {

      float accum = 0.0f;

      for (int k = 0; k < ifeat; k++) {

        float w = W[idx_col_major(k, row, ifeat)];
        float x = X[idx_col_major(k, col, ifeat)];

        accum += w * x;
      }

      C[idx_col_major(row, col, ofeat)] = accum;
    }
  }
}

// -----------------------------------------------------------------------------
// Bias add
//
// input shape: [features x batch_size]
// bias shape: [features]
// -----------------------------------------------------------------------------
void bias_add(
    scalar_t* X,
    const scalar_t* bias,
    int batch_size,
    int features)
{
    for (int col = 0; col < batch_size; col++) {
  for (int row = 0; row < features; row++) {

      X[idx_col_major(row, col, features)] += bias[row];
      //X[row * batch_size + col] += bias[row];
    }
  }
}

// -----------------------------------------------------------------------------
// Optional ReLU
// -----------------------------------------------------------------------------
void relu(
    scalar_t* X,
    int batch_size,
    int features)
{
  for (int col = 0; col < batch_size; col++) {
    for (int row = 0; row < features; row++) {

      float& v = X[idx_col_major(row, col, features)];

      if (v < 0.0f)
        v = 0.0f;
    }
  }
}

// -----------------------------------------------------------------------------
// forward pass
// -----------------------------------------------------------------------------
int mlp_fp_cpu(
    scalar_t* X,
    int input_features,
    int batch_size,
    std::vector<scalar_t*>& WPtr,
    int num_layers,
    std::vector<int>& output_features,
    std::vector<scalar_t*>& BPtr,
    scalar_t* Y,
    std::vector<scalar_t>& reserved_space,
    int use_bias,
    int use_relu)
{
  scalar_t* input = nullptr;
  scalar_t* output = nullptr;

  size_t reserved_offset = 0;

  for (int layer = 0; layer < num_layers; layer++) {

    scalar_t* weight = WPtr[layer];
    scalar_t* bias   = use_bias ? BPtr[layer] : nullptr;

    int ifeat = (layer == 0) ? input_features : output_features[layer - 1];

    int ofeat = output_features[layer];

    // Input selection
    if (layer == 0) {
      input = X;
    } else {
      int prev_ofeat = output_features[layer - 1];
      input = reserved_space.data()
        + reserved_offset
        - prev_ofeat * batch_size;
    }

    // Output selection
    if (layer == num_layers - 1) {
      output = Y;
    } else {
      output = reserved_space.data() + reserved_offset;
    }

    // GEMM
    gemm_w_transpose(
        weight,
        input,
        output,
        ifeat,
        ofeat,
        batch_size);

    // Bias
    if (use_bias) {
      bias_add(output, bias, batch_size, ofeat);
    }

    // Relu
    if (use_relu) {
      relu(output, batch_size, ofeat);
    }

    // Advance reserved-space pointer
    if (layer != num_layers - 1) {
      reserved_offset += ofeat * batch_size;
    }
  }

  return 0;
}

// -----------------------------------------------------------------------------
// Allocate random weights
// -----------------------------------------------------------------------------
void get_weight_space(
    std::vector<scalar_t*>& w_ptr,
    int input_feature,
    int num_layers,
    std::vector<int>& output_features)
{
#ifdef DEBUG
  printf("W\n");
#endif
  for (int l = 0; l < num_layers; l++) {

    int size;

    if (l == 0)
      size = input_feature * output_features[l];
    else
      size = output_features[l - 1] * output_features[l];

    scalar_t* w = new scalar_t[size];

    for (int i = 0; i < size; i++) {
      w[i] = distr(g);
#ifdef DEBUG
      printf("%f ", w[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif

    w_ptr.push_back(w);
  }
}

// -----------------------------------------------------------------------------
// Allocate random biases
// -----------------------------------------------------------------------------
void get_bias_space(
    std::vector<scalar_t*>& b_ptr,
    int num_layers,
    std::vector<int>& output_features)
{
#ifdef DEBUG
  printf("Bias\n");
#endif
  for (int l = 0; l < num_layers; l++) {

    int size = output_features[l];

    scalar_t* b = new scalar_t[size];

    for (int i = 0; i < size; i++) {
      b[i] = distr(g);
#ifdef DEBUG
      printf("%f ", b[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif

    b_ptr.push_back(b);
  }
}

scalar_t* reference(
    const int num_layers,
    const int batch_size,
    const int input_features,
    const int hidden_dim,
    const int num_outputs,
    const int use_relu,
    const int use_bias,
    const int repeat)
{
  g.seed(123);

  std::vector<int> output_features(num_layers);

  for (int l = 0; l < num_layers - 1; l++)
    output_features[l] = hidden_dim;

  output_features[num_layers - 1] = num_outputs;

  // -------------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------------
  size_t input_size = batch_size * input_features;

  scalar_t* input = new scalar_t[input_size];

#ifdef DEBUG
  printf("Input\n");
#endif
  for (size_t i = 0; i < input_size; i++) {
    input[i] = distr(g);
#ifdef DEBUG
    printf("%f ", input[i]);
#endif
  }
#ifdef DEBUG
  printf("\n");
#endif

  // -------------------------------------------------------------------------
  // Weights and biases
  // -------------------------------------------------------------------------
  std::vector<scalar_t*> w_ptr;
  std::vector<scalar_t*> b_ptr;

  get_weight_space(
      w_ptr,
      input_features,
      num_layers,
      output_features);

  get_bias_space(
      b_ptr,
      num_layers,
      output_features);

  // -------------------------------------------------------------------------
  // Output
  // -------------------------------------------------------------------------
  scalar_t* output = new scalar_t[batch_size * num_outputs];

  // -------------------------------------------------------------------------
  // Reserved space
  // -------------------------------------------------------------------------
  size_t reserved_elems = 0;

  for (int l = 0; l < num_layers - 1; l++)
    reserved_elems += output_features[l] * batch_size;

  std::vector<scalar_t> reserved_space(reserved_elems);

  //long long total_ns = 0;
  //auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    mlp_fp_cpu(
        input,
        input_features,
        batch_size,
        w_ptr,
        num_layers,
        output_features,
        b_ptr,
        output,
        reserved_space,
        use_bias,
        use_relu);
  }

  //auto end = std::chrono::steady_clock::now();
  //total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  //printf("Average CPU execution time: %lf us\n", (total_ns * 1e-3) / repeat);

  delete[] input;

  for (auto p : w_ptr) delete[] p;
  for (auto p : b_ptr) delete[] p;

  return output;
}
