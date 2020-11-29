#include <cstdio>
#include <hip/hip_runtime.h>

#define TILE_SIZE 5900
#define NTHREADS 256

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int ITER = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.0 / d2, 1.0 / d3, 1.0 / d1};
static const float shape_input_r[] = {1.0 / d4, 1.0 / d5, 1.0 / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("Failed!\n");
      exit(-1);
    }
  }
}

  __global__
void tensor_transpose(int dim_input, 
    int dim_output, 
    int nblocks, 
    int tile_size,
    int *shape_input, 
    int *shape_output, 
    float *shape_input_r, 
    float *shape_output_r, 
    int *stride_input,
    int *stride_output_local, 
    int *stride_output_global,
    double *input, 
    double *output) 
{
  __shared__ double tile[TILE_SIZE];

  for (int block_idx = blockIdx.x; block_idx < nblocks; block_idx += gridDim.x) {
    int it = block_idx, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it * shape_input_r[i];
      offset1 += stride_input[i] * (it - im * shape_input[i]);
      it = im;
    }

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      tile[i] = input[i + block_idx * tile_size];
    }

    __syncthreads();

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
      it = i;
      int offset2 = 0, local_offset = 0;
      for (int j = 0; j < dim_output; j++) {
        im = it * shape_output_r[j];
        int tmp = it - im * shape_output[j];
        offset2 += stride_output_global[j] * tmp;
        local_offset += stride_output_local[j] * tmp;
        it = im;
      }
      output[offset1 + offset2] = tile[local_offset];
    }

    __syncthreads();
  }
}

int main(int argv, char **argc) {
  if (argv > 1) {
    ITER = atoi(argc[1]);
  }

  double *input = new double[data_size]();
  double *output = new double[data_size]();

  for (size_t i = 0; i < data_size; i++) {
    input[i] = i;
  }

  const int nblocks = d4 * d5 * d6;
  const int tile_size = d1 * d2 * d3;
  const int dim_output = 3;
  const int dim_input = 3;
  double *device_output, *device_input;
  int *device_shape_input, *device_shape_output;
  float *device_shape_input_r, *device_shape_output_r;
  int *device_stride_output_local, *device_stride_output_global;
  int *device_stride_input;

  hipMalloc(&device_output, data_size * sizeof(double));
  hipMalloc(&device_input, data_size * sizeof(double));
  hipMalloc(&device_shape_input, dim_input * sizeof(int));
  hipMalloc(&device_shape_input_r, dim_input * sizeof(float));
  hipMalloc(&device_shape_output, dim_output * sizeof(int));
  hipMalloc(&device_shape_output_r, dim_output * sizeof(float));
  hipMalloc(&device_stride_input, dim_input * sizeof(int));
  hipMalloc(&device_stride_output_local, dim_output * sizeof(int));
  hipMalloc(&device_stride_output_global, dim_output * sizeof(int));

  hipMemcpy(device_input, input, 
      data_size * sizeof(double), hipMemcpyHostToDevice );
  hipMemcpy(device_shape_input, shape_input, 
      dim_input * sizeof(int), hipMemcpyHostToDevice );
  hipMemcpy(device_shape_input_r, shape_input_r, 
      dim_input * sizeof(float), hipMemcpyHostToDevice );
  hipMemcpy(device_shape_output, shape_output, 
      dim_output * sizeof(int), hipMemcpyHostToDevice );
  hipMemcpy(device_shape_output_r, shape_output_r, 
      dim_output * sizeof(float), hipMemcpyHostToDevice );
  hipMemcpy(device_stride_input, stride_input, 
      dim_input * sizeof(int), hipMemcpyHostToDevice );
  hipMemcpy(device_stride_output_local, stride_output_local, 
      dim_output * sizeof(int), hipMemcpyHostToDevice );
  hipMemcpy(device_stride_output_global, stride_output_global, 
      dim_output * sizeof(int), hipMemcpyHostToDevice );


  for (size_t i = 0; i < ITER; ++i) {
    hipLaunchKernelGGL(tensor_transpose, dim3(nblocks), dim3(NTHREADS), 0, 0, dim_input, dim_output, nblocks, tile_size,
        device_shape_input, 
        device_shape_output,
        device_shape_input_r,
        device_shape_output_r,
        device_stride_input,
        device_stride_output_local,
        device_stride_output_global,
        device_input, device_output);
  }

  hipMemcpy(output, device_output, data_size * sizeof(double), hipMemcpyDeviceToHost);

  hipFree(device_output);
  hipFree(device_input);
  hipFree(device_shape_input);
  hipFree(device_shape_input_r);
  hipFree(device_shape_output);
  hipFree(device_shape_output_r);
  hipFree(device_stride_input);
  hipFree(device_stride_output_local);
  hipFree(device_stride_output_global);

  verify(input, output);

  delete [] input;
  delete [] output;

  return 0;
}
