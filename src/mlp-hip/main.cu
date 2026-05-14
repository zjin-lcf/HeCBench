#include <cassert>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include "mlp.h"
#include "reference.h"

// Returns the reserved space (in elements) needed for the MLP
template <typename T>
size_t get_mlp_reserved_space(int64_t batch_size, int num_layers,
                              std::vector<int> &output_features)
{
  size_t res_space = 0;
  // Need to store output of every intermediate MLP - size equal to output_features[i] * batch_size
  for (int l = 0; l < num_layers; l++) {
    res_space += output_features[l] * batch_size;
  }
  return res_space * sizeof(T);
}

template <typename T>
void free_mlp_space(std::vector<T*> &w_ptr, int num_layers)
{
  for (int l = 0; l < num_layers; l++)
    GPU_CHECK(hipFree(w_ptr[l]));
}

template <typename T>
void
get_mlp_weight_space(std::vector<T*> &w_ptr, int input_feature, int num_layers, std::vector<int> &output_features)
{
  T *t, *w;
  int size;
#ifdef DEBUG
  printf("W\n");
#endif
  for (int l = 0; l < num_layers; l++) {
    if (l == 0) {
      size = input_feature * output_features[l];
      t = (T*) malloc (sizeof(T) * size);
      for (int i = 0; i < size; i++) {
        t[i] = distr(g);
#ifdef DEBUG
        printf("%f ", t[i]);
#endif
      }
#ifdef DEBUG
      printf("\n");
#endif
      GPU_CHECK(hipMalloc(&w, sizeof(T) * size));
      GPU_CHECK(hipMemcpy(w, t, sizeof(T) * size, hipMemcpyHostToDevice));
      w_ptr.push_back(w);
    }
    else {
      size = output_features[l] * output_features[l-1];
      t = (T*) malloc (sizeof(T) * size);
      for (int i = 0; i < size; i++) {
        t[i] = distr(g);
#ifdef DEBUG
        printf("%f ", t[i]);
#endif
      }
#ifdef DEBUG
      printf("\n");
#endif
      GPU_CHECK(hipMalloc(&w, sizeof(T) * size));
      GPU_CHECK(hipMemcpy(w, t, sizeof(T) * size, hipMemcpyHostToDevice));
      w_ptr.push_back(w);
    }
    free(t);
  }
}

template <typename T>
void
get_mlp_bias_space(std::vector<T*> &w_ptr, int num_layers, std::vector<int> &output_features)
{
  T *t, *w;
  int size;
#ifdef DEBUG
  printf("Bias\n");
#endif
  for (int l = 0; l < num_layers; l++) {
    size = output_features[l];
    t = (T*) malloc (sizeof(T) * size);
    for (int i = 0; i < size; i++) {
      t[i] = distr(g);
#ifdef DEBUG
      printf("%f ", t[i]);
#endif
    }
#ifdef DEBUG
    printf("\n");
#endif
    GPU_CHECK(hipMalloc(&w, sizeof(T) * size));
    GPU_CHECK(hipMemcpy(w, t, sizeof(T) * size, hipMemcpyHostToDevice));
    w_ptr.push_back(w);
    free(t);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 9) {
    printf("Usage: %s ", argv[0]);
    printf("<number of layers> <batch size> <input feature size> ");
    printf("<hidden dimension> <number of outputs> <use relu> <use bias> <repeat>\n");
    return 1;
  }
  const int num_layers = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int input_features = atoi(argv[3]);
  const int hidden_dim = atoi(argv[4]);
  const int num_outputs = atoi(argv[5]);
  const int use_relu = atoi(argv[6]);
  const int use_bias = atoi(argv[7]);
  const int repeat = atoi(argv[8]);

  typedef float scalar_t; // FP16 MLP

  g.seed(123);
  std::vector<int> output_features(num_layers);
  for (int l = 0; l < num_layers - 1; l++) {
    output_features[l] = hidden_dim;
  }
  output_features[num_layers-1] = num_outputs;

  scalar_t *input, *d_input;
  size_t in_size = sizeof(scalar_t) * batch_size * input_features;
  GPU_CHECK(hipMalloc(&d_input, in_size));

  input = (scalar_t*) malloc (in_size);
#ifdef DEBUG
  printf("Input\n");
#endif
  for (int i = 0; i < batch_size * input_features; i++) {
    input[i] = distr(g);
#ifdef DEBUG
    printf("%f ", input[i]);
#endif
  }
#ifdef DEBUG
  printf("\n");
#endif

  std::vector<scalar_t*> w_ptr;
  get_mlp_weight_space(w_ptr, input_features, num_layers, output_features);

  std::vector<scalar_t*> b_ptr;
  get_mlp_bias_space(b_ptr, num_layers, output_features);

  // create output/workspace tensor
  auto out_size = batch_size * num_outputs * sizeof(scalar_t);
  scalar_t *out, *d_out;
  GPU_CHECK(hipMalloc(&d_out, out_size));
  out = (scalar_t *) malloc (out_size);

  scalar_t *reserved_space;
  auto reserved_size = get_mlp_reserved_space<scalar_t>(batch_size, num_layers, output_features);
  GPU_CHECK(hipMalloc(&reserved_space, reserved_size));

  // allocate fixed 4MB workspace for hipblaslt for now, and this gets at least 4 MB
  auto lt_workspace_size = 4 * 1024 * 1024 * sizeof(scalar_t);
  scalar_t *lt_workspace;
  GPU_CHECK(hipMalloc(&lt_workspace, lt_workspace_size));

  // cuBLASLt handle (global variable)
  assert(hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS);

  long time = 0;

  int error = 0;

  GPU_CHECK(hipMemcpy(d_input, input, in_size, hipMemcpyHostToDevice));

  // warmup required
  for (int i = 0; i < 300; i++) {

    error = mlp_fp<scalar_t>(
        d_input,
        input_features,
        batch_size,
        w_ptr.data(),
        num_layers,
        output_features.data(),
        b_ptr.data(),
        d_out,
        reserved_space,
        use_bias,
        use_relu,
        (void*) (lt_workspace));

    GPU_CHECK(hipDeviceSynchronize());

    if (error) {
      printf("MLP execution failed.\n");
      break;
    }
  }

  if (error == 0) {
    GPU_CHECK(hipMemcpy(out, d_out, out_size, hipMemcpyDeviceToHost));

    float *out_r = reference(
      num_layers, batch_size, input_features,
      hidden_dim, num_outputs, use_relu, use_bias, repeat);

    bool ok = true;
    for (int i = 0; i < num_outputs; i++) {
      for (int j = 0; j < batch_size; j++)
        if (std::fabs((float)out[j * num_outputs + i] -
                      (float)out_r[j * num_outputs + i]) > 1e-2f) {
          printf("%4d : %f %f\n", i, (float)out[j * num_outputs + i], out_r[j * num_outputs + i]);
          ok = false;
          break;
        }
#ifdef DEBUG
        else {
          printf("%4d : %f %f\n", i, (float)out[j * num_outputs + i], out_r[j * num_outputs + i]);
        }
#endif
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    // benchmarking
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      mlp_fp<scalar_t>(
          d_input,
          input_features,
          batch_size,
          w_ptr.data(),
          num_layers,
          output_features.data(),
          b_ptr.data(),
          d_out,
          reserved_space,
          use_bias,
          use_relu,
          (void*) (lt_workspace));
    }
    GPU_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of MLP: %lf (us)\n", (time * 1e-3) / repeat);
  }

  free(input);
  free(out);
  GPU_CHECK(hipFree(d_input));
  GPU_CHECK(hipFree(d_out));
  GPU_CHECK(hipFree(reserved_space));
  GPU_CHECK(hipFree(lt_workspace));
  free_mlp_space(w_ptr, num_layers);
  free_mlp_space(b_ptr, num_layers);
  assert(hipblasLtDestroy(handle) == HIPBLAS_STATUS_SUCCESS);
  return error;
}
