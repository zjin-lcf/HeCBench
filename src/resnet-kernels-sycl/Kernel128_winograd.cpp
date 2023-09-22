#include "util.h"

const char inputName128[] = "data/input_14_1_128.bin";
const char biasName128[] = "data/bias_128.bin";
const char weight_winograd_Name128[] = "data/weight_winograd_128_128.bin";
const char bnBias_winograd_Name128[] = "data/bnBias_winograd_128.bin";
const char bnScale_winograd_Name128[] = "data/bnScale_winograd_128.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )
#define __syncthreads() item.barrier(sycl::access::fence_space::local_space)

void kernel_128_winograd_BtdB(
  sycl::nd_item<2> &item,
        float *__restrict input,
  const float *__restrict pInputs,
        float *__restrict pOutputs)
{
  int Inx = item.get_group(1)<<2,
      Iny0 = item.get_group(0)<<2,
      Iny1 = item.get_local_id(0),
      Inz = item.get_local_id(1);
  int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;

  int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
  }
  __syncthreads();

  float BTd[6];
  switch(Iny1) {
    case 0:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
      }
      break;
    case 1:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 2:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
      }
      break;
    case 3:
      for (int j = 0; j < 6; j++) {
        BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 4:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
      }
      break;
    case 5:
      for (int j = 0; j < 6; j++) {
        BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
      }
      break;
  }
  __syncthreads();

  int tmp_offset = Iny1*768+Inz;
  for (int i = 0; i < 6; i++) {
    input[tmp_offset + i*stride_c] = BTd[i];
  }
  __syncthreads();

  float BTdB[6];
  switch(Iny1) {
    case 0:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 1:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 2:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 3:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 4:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
      }
      break;
    case 5:
      for (int i = 0; i < 6; i++) {
        BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
      }
      break;
  }
  __syncthreads();

  for (int i = 0; i < 6; i++) {
    pOutputs[(Iny1 + i*6)*2048 +
             (item.get_group(1)*4+item.get_group(0))*128 + Inz] = BTdB[i];
  }
}

void kernel_128_winograd_AtIA(
  sycl::nd_item<3> &item,
        float &bias,
        float &scale,
        float *__restrict input,
  const float *__restrict pInputs,
  const float *__restrict pBiases,
  const float *__restrict pScales,
        float *__restrict pOutputs)
{
  int Tilex = item.get_group(2), Tiley = item.get_group(1),
      Iny = item.get_local_id(1), kz = item.get_group(0), Inx = item.get_local_id(2);
  int c_input = Inx*6 + Iny;

  input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
  bias = pBiases[kz];
  scale = pScales[kz];
  __syncthreads();

  float tmp = 0;
  switch(Inx) {
    case 0:
      tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
      break;
    case 1:
      tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
      break;
    case 2:
      tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
      break;
    case 3:
      tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
      break;
  }
  __syncthreads();

  input[c_input] = tmp;
  __syncthreads();

  if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;

  int x;
  float o;
  switch(Iny) {
    case 0:
      x = Inx*6;
      o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4])+ bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*128 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*128 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*128 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*128 + kz] = o > 0 ? o : 0;
      break;
  }
}

void kernel_128_OuterProduct_128(
  sycl::nd_item<2> &item,
        float *__restrict input,
  const float *__restrict A,
  const float *__restrict B,
        float *__restrict C)
{
  int Tile = item.get_group(1), Part = item.get_group(0),
      tX = item.get_local_id(1), tY = item.get_local_id(0);
  int c_input = tY*128 + tX, c_kernel = c_input;
  int T_offset = (Tile<<11) + (Part<<10) + c_input;
  int B_offset = (Tile<<14) + c_kernel;

  float *kernel = input + 1024, *out = kernel + 8192;
  int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896,
                      1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
                      2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944,
                      3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};
  out[c_input] = 0.0f;

  input[c_input] = A[T_offset];

  for (int k = 0; k < 4; k++) {
    int B_start = B_offset + (k<<12); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    __syncthreads();

    float sum = 0;
    int y_tmp = (tY<<7)+(k<<5);
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
    }
    out[tY*128 + tX] += sum;
    __syncthreads();
  }

  C[T_offset] = out[c_input];
}

void kernel_128(sycl::queue &q, double &time, double &ktime) {
  float *input_ = get_parameter(inputName128, 16*16*128);
  float *bias = get_parameter(biasName128, 128);

  float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);
  float *bnBias, *bnScale;

  int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nBias = 128,
      nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128;

  bnBias = get_parameter(bnBias_winograd_Name128, 128);
  bnScale = get_parameter(bnScale_winograd_Name128, 128);

  float result[nOutput];

  auto start = std::chrono::steady_clock::now();

  float *input = sycl::malloc_device<float>(nInput, q);
  q.memcpy(input, input_, sizeof(float) * nInput);

  float *l_weights = sycl::malloc_device<float>(nWeights, q);
  q.memcpy(l_weights, kernel, sizeof(float) * nWeights);

  float *output = sycl::malloc_device<float>(nOutput, q);

  float *t_input = sycl::malloc_device<float>(nTransInput, q);

  float *ip = sycl::malloc_device<float>(nInnerProd, q);

  q.memset(output, 0, sizeof(float) * nOutput);
  q.memset(t_input, 0, sizeof(float) * nTransInput);
  q.memset(ip, 0, sizeof(float) * nInnerProd);

  float *l_bnBias = sycl::malloc_device<float>(nBias, q);
  q.memcpy(l_bnBias, bnBias, sizeof(float) * nBias);

  float *l_bnScale = sycl::malloc_device<float>(nBias, q);
  q.memcpy(l_bnScale, bnScale, sizeof(float) * nBias);

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  sycl::range<2> gws (4*6, 4*128);
  sycl::range<2> lws (6, 128);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sm (sycl::range<1>(6*6*128), cgh);
    cgh.parallel_for<class k1>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      kernel_128_winograd_BtdB (item, sm.get_pointer(), input, t_input);
    });
  });

  sycl::range<2> gws2 (2*8, 36*128);
  sycl::range<2> lws2 (8, 128);
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sm (sycl::range<1>(8*128 + 64*128 + 8*128), cgh);
    cgh.parallel_for<class k2>(sycl::nd_range<2>(gws2, lws2), [=] (sycl::nd_item<2> item) {
      kernel_128_OuterProduct_128(item, sm.get_pointer(), t_input, l_weights, ip);
    });
  });

  sycl::range<3> gws3 (128, 4*6, 4*6);
  sycl::range<3> lws3 (1, 6, 6);
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 0> sm_bias (cgh);
    sycl::local_accessor<float, 0> sm_scale (cgh);
    sycl::local_accessor<float, 1> sm_input (sycl::range<1>(6*6), cgh);
    cgh.parallel_for<class k3>(sycl::nd_range<3>(gws3, lws3), [=] (sycl::nd_item<3> item) {
      kernel_128_winograd_AtIA(item,
        sm_bias, sm_scale, sm_input.get_pointer(),
        ip, l_bnBias, l_bnScale, output);
    });
  });

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();

  q.memcpy(result, output, sizeof(float) * nOutput).wait();

  sycl::free(input, q);
  sycl::free(t_input, q);
  sycl::free(l_weights, q);
  sycl::free(l_bnBias, q);
  sycl::free(l_bnScale, q);
  sycl::free(ip, q);
  sycl::free(output, q);

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  #ifdef DEBUG
  double s = 0;
  for (int i = 0; i < nOutput; i++) {
    s += result[i];
  }
  printf("Check sum: %lf\n", s);
  #endif

  free(kernel);
  free(bnScale);
  free(bnBias);
  free(bias);
  free(input_);
}
