#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

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
    double *output,
    sycl::nd_item<3> item_ct1,
    double *tile) 
{

  for (int block_idx = item_ct1.get_group(2); block_idx < nblocks;
       block_idx += item_ct1.get_group_range(2)) {
    int it = block_idx, im = 0, offset1 = 0;
    for (int i = 0; i < dim_input; i++) {
      im = it * shape_input_r[i];
      offset1 += stride_input[i] * (it - im * shape_input[i]);
      it = im;
    }

    for (int i = item_ct1.get_local_id(2); i < tile_size;
         i += item_ct1.get_local_range().get(2)) {
      tile[i] = input[i + block_idx * tile_size];
    }

    item_ct1.barrier();

    for (int i = item_ct1.get_local_id(2); i < tile_size;
         i += item_ct1.get_local_range().get(2)) {
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

    item_ct1.barrier();
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

  dpct::dpct_malloc(&device_output, data_size * sizeof(double));
  dpct::dpct_malloc(&device_input, data_size * sizeof(double));
  dpct::dpct_malloc(&device_shape_input, dim_input * sizeof(int));
  dpct::dpct_malloc(&device_shape_input_r, dim_input * sizeof(float));
  dpct::dpct_malloc(&device_shape_output, dim_output * sizeof(int));
  dpct::dpct_malloc(&device_shape_output_r, dim_output * sizeof(float));
  dpct::dpct_malloc(&device_stride_input, dim_input * sizeof(int));
  dpct::dpct_malloc(&device_stride_output_local, dim_output * sizeof(int));
  dpct::dpct_malloc(&device_stride_output_global, dim_output * sizeof(int));

  dpct::dpct_memcpy(device_input, input, data_size * sizeof(double),
                    dpct::host_to_device);
  dpct::dpct_memcpy(device_shape_input, shape_input, dim_input * sizeof(int),
                    dpct::host_to_device);
  dpct::dpct_memcpy(device_shape_input_r, shape_input_r,
                    dim_input * sizeof(float), dpct::host_to_device);
  dpct::dpct_memcpy(device_shape_output, shape_output, dim_output * sizeof(int),
                    dpct::host_to_device);
  dpct::dpct_memcpy(device_shape_output_r, shape_output_r,
                    dim_output * sizeof(float), dpct::host_to_device);
  dpct::dpct_memcpy(device_stride_input, stride_input, dim_input * sizeof(int),
                    dpct::host_to_device);
  dpct::dpct_memcpy(device_stride_output_local, stride_output_local,
                    dim_output * sizeof(int), dpct::host_to_device);
  dpct::dpct_memcpy(device_stride_output_global, stride_output_global,
                    dim_output * sizeof(int), dpct::host_to_device);

  for (size_t i = 0; i < ITER; ++i) {
    dpct::buffer_t device_shape_input_buf_ct4 =
        dpct::get_buffer(device_shape_input);
    dpct::buffer_t device_shape_output_buf_ct5 =
        dpct::get_buffer(device_shape_output);
    dpct::buffer_t device_shape_input_r_buf_ct6 =
        dpct::get_buffer(device_shape_input_r);
    dpct::buffer_t device_shape_output_r_buf_ct7 =
        dpct::get_buffer(device_shape_output_r);
    dpct::buffer_t device_stride_input_buf_ct8 =
        dpct::get_buffer(device_stride_input);
    dpct::buffer_t device_stride_output_local_buf_ct9 =
        dpct::get_buffer(device_stride_output_local);
    dpct::buffer_t device_stride_output_global_buf_ct10 =
        dpct::get_buffer(device_stride_output_global);
    dpct::buffer_t device_input_buf_ct11 = dpct::get_buffer(device_input);
    dpct::buffer_t device_output_buf_ct12 = dpct::get_buffer(device_output);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<double, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          tile_acc_ct1(sycl::range<1>(5900 /*TILE_SIZE*/), cgh);
      auto device_shape_input_acc_ct4 =
          device_shape_input_buf_ct4.get_access<sycl::access::mode::read_write>(
              cgh);
      auto device_shape_output_acc_ct5 =
          device_shape_output_buf_ct5
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_shape_input_r_acc_ct6 =
          device_shape_input_r_buf_ct6
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_shape_output_r_acc_ct7 =
          device_shape_output_r_buf_ct7
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_stride_input_acc_ct8 =
          device_stride_input_buf_ct8
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_stride_output_local_acc_ct9 =
          device_stride_output_local_buf_ct9
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_stride_output_global_acc_ct10 =
          device_stride_output_global_buf_ct10
              .get_access<sycl::access::mode::read_write>(cgh);
      auto device_input_acc_ct11 =
          device_input_buf_ct11.get_access<sycl::access::mode::read_write>(cgh);
      auto device_output_acc_ct12 =
          device_output_buf_ct12.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                             sycl::range<3>(1, 1, NTHREADS),
                                         sycl::range<3>(1, 1, NTHREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         tensor_transpose(
                             dim_input, dim_output, nblocks, tile_size,
                             (int *)(&device_shape_input_acc_ct4[0]),
                             (int *)(&device_shape_output_acc_ct5[0]),
                             (float *)(&device_shape_input_r_acc_ct6[0]),
                             (float *)(&device_shape_output_r_acc_ct7[0]),
                             (int *)(&device_stride_input_acc_ct8[0]),
                             (int *)(&device_stride_output_local_acc_ct9[0]),
                             (int *)(&device_stride_output_global_acc_ct10[0]),
                             (double *)(&device_input_acc_ct11[0]),
                             (double *)(&device_output_acc_ct12[0]), item_ct1,
                             tile_acc_ct1.get_pointer());
                       });
    });
  }

  dpct::dpct_memcpy(output, device_output, data_size * sizeof(double),
                    dpct::device_to_host);

  dpct::dpct_free(device_output);
  dpct::dpct_free(device_input);
  dpct::dpct_free(device_shape_input);
  dpct::dpct_free(device_shape_input_r);
  dpct::dpct_free(device_shape_output);
  dpct::dpct_free(device_shape_output_r);
  dpct::dpct_free(device_stride_input);
  dpct::dpct_free(device_stride_output_local);
  dpct::dpct_free(device_stride_output_global);

  verify(input, output);

  delete [] input;
  delete [] output;

  return 0;
}
