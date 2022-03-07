#include "util.h"

const char inputName256[] = "data/input_14_1_256.bin";
const char weight_winograd_Name256[] = "data/weight_winograd_256_256.bin";
const char bnBias_winograd_Name256[] = "data/bnBias_winograd_256.bin";
const char bnScale_winograd_Name256[] = "data/bnScale_winograd_256.bin";

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )
#define __syncthreads() item.barrier(access::fence_space::local_space)

void kernel_256_winograd_BtdB(
  nd_item<3> &item,
        float *__restrict input,
  const float *__restrict pInputs,
        float *__restrict pOutputs)
{
  int Inx = item.get_group(2)<<2, Iny0 = item.get_group(1)<<2, Part = item.get_group(0),
      Iny1 = item.get_local_id(1), Inz = item.get_local_id(2);
  int Iny = Iny0+Iny1, stride_r = 4096, stride_c = 256; // 4096 = 16*256
  int c_glb_start = Inx*stride_r + Iny*stride_c + Inz + (Part<<7), c_input = Iny1*128 + Inz;

  int stride_768[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
  for (int i = 0; i < 6; i++) {
    input[c_input + stride_768[i]] = pInputs[c_glb_start + i*stride_r];
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
    input[tmp_offset + i*128] = BTd[i];
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
    pOutputs[(Iny1 + i*6)*4096 + 
             (item.get_group(2)*4+item.get_group(1))*256 + 
             Inz + (Part<<7)] = BTdB[i];
  }
}

void kernel_256_winograd_AtIA(
  nd_item<3> &item,
        float *__restrict bias,
        float *__restrict scale,
        float *__restrict input,
  const float *__restrict pInputs,
  const float *__restrict pBiases,
  const float *__restrict pScales,
        float *__restrict pOutputs)
{
  int Tilex = item.get_group(2), Tiley = item.get_group(1),
      Iny = item.get_local_id(1), kz = item.get_group(0), Inx = item.get_local_id(2);
  int c_input = Inx*6 + Iny;

  input[c_input] = pInputs[c_input*16*256 + (Tilex*4+Tiley)*256 + kz];
  *bias = pBiases[kz];
  *scale = pScales[kz];
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
      o = *scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4]) + *bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*256 + kz] = o > 0 ? o : 0;
      break;
    case 1:
      x = Inx*6;
      o = *scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + *bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*256 + kz] = o > 0 ? o : 0;
      break;
    case 2:
      if (Tiley == 3) break;
      x = Inx*6;
      o = *scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + *bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*256 + kz] = o > 0 ? o : 0;
      break;
    case 3:
      if (Tiley == 3) break;
      x = Inx*6;
      o = *scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + *bias;
      pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*256 + kz] = o > 0 ? o : 0;
      break;
  }
}

void kernel_256_OuterProduct_256(
  nd_item<2> &item,
        float *__restrict input,
  const float *__restrict A,
  const float *__restrict B,
        float *__restrict C)
{
  int Tile = item.get_group(1), Part = item.get_group(0),
      tX = item.get_local_id(1), tY = item.get_local_id(0);
  int c_input = tY*256 + tX,
      c_kernel = c_input,
      T_offset = (Tile<<12) + (Part<<11) + c_input, B_offset = (Tile<<16) + c_kernel;

  float *kernel = input + 2048, *out = kernel + 8192;
  int B_stride[32] = {0, 256, 512, 768, 1024, 1280, 1536, 1792, 
                      2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,
                      4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888,
                      6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936};
  out[c_input] = 0.0f;
  out[c_input+1024] = 0;

  input[c_input] = A[T_offset];
  input[c_input+1024] = A[T_offset+1024];

  for (int k = 0; k < 8; k++) {
    int B_start = B_offset + (k<<13); // 32*64
    kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
    kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
    kernel[c_kernel+4096] = B[B_start+4096], kernel[c_kernel+5120] = B[B_start+5120];
    kernel[c_kernel+6144] = B[B_start+6144], kernel[c_kernel+7168] = B[B_start+7168];

    __syncthreads();

    float sum = 0, sum1 = 0;
    int y_tmp = (tY<<8)+(k<<5), y_tmp1 = y_tmp+1024;
    for (int j = 0; j < 32; j++) {
      sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
      sum1 += input[y_tmp1 + j] * kernel[tX + B_stride[j]];
    }
    out[c_input] += sum;
    out[c_input+1024] += sum1;
    __syncthreads();
  }

  C[T_offset] = out[c_input];
  C[T_offset+1024] = out[c_input+1024];
}

int kernel_256(queue &q) {
  float *input_ = get_parameter(inputName256, 16*16*256);
  uint64_t nT1 = 0, nT2 = 0;

  float *kernel = get_parameter(weight_winograd_Name256, 36*256*256);
  int nInput = 16*16*256, nOutput = 16*16*256, nWeights = 36*256*256, nBias = 256,
      nTransInput = 16*6*6*256, nInnerProd = 16*6*6*256;
  float *bnBias, *bnScale;

  bnBias = get_parameter(bnBias_winograd_Name256, 256);
  bnScale = get_parameter(bnScale_winograd_Name256, 256);

  float result[nOutput];

  nT1 = getTimeMicroseconds64();

  buffer<float, 1> input(input_, nInput);
  buffer<float, 1> l_weights(kernel, nWeights);
  buffer<float, 1> output(nOutput);
  buffer<float, 1> t_input(nTransInput);
  buffer<float, 1> ip(nInnerProd);

  q.submit([&] (handler &cgh) {
    auto acc = output.get_access<sycl_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = t_input.get_access<sycl_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  q.submit([&] (handler &cgh) {
    auto acc = ip.get_access<sycl_write>(cgh);
    cgh.fill(acc, 0.f);
  });

  buffer<float, 1> l_bnBias (bnBias, nBias);
  buffer<float, 1> l_bnScale (bnScale, nBias);

  //kernel_256_winograd_BtdB <<<dim3(4, 4, 2), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
  //kernel_256_OuterProduct_256<<<dim3(36, 2), dim3(256, 4), (8*256 + 32*256 + 8*256)<<2 >>> (t_input, l_weights, ip);
  //kernel_256_winograd_AtIA <<<dim3(4, 4, 256), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);

  range<3> gws (2, 4*6, 4*128);
  range<3> lws (1, 6, 128);
  q.submit([&] (handler &cgh) {
    auto i = input.get_access<sycl_read>(cgh);
    auto t = t_input.get_access<sycl_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sm (6*6*128, cgh);
    cgh.parallel_for<class k1>(nd_range<3>(gws, lws), [=] (nd_item<3> item) {
      kernel_256_winograd_BtdB (item, sm.get_pointer(), i.get_pointer(), t.get_pointer());
    });
  });

  range<2> gws2 (2*4, 36*256);
  range<2> lws2 (4, 256);
  q.submit([&] (handler &cgh) {
    auto t = t_input.get_access<sycl_read>(cgh);
    auto w = l_weights.get_access<sycl_read>(cgh);
    auto i = ip.get_access<sycl_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sm (8*256 + 32*256 + 8*256, cgh);
    cgh.parallel_for<class k2>(nd_range<2>(gws2, lws2), [=] (nd_item<2> item) {
      kernel_256_OuterProduct_256(item, sm.get_pointer(),
        t.get_pointer(), w.get_pointer(), i.get_pointer());
    });
  });

  range<3> gws3 (256, 4*6, 4*6);
  range<3> lws3 (1, 6, 6);
  q.submit([&] (handler &cgh) {
    auto i = ip.get_access<sycl_read>(cgh);
    auto b = l_bnBias.get_access<sycl_read>(cgh);
    auto s = l_bnScale.get_access<sycl_read>(cgh);
    auto o = output.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sm_bias (1, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sm_scale (1, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sm_input (6*6, cgh);
    cgh.parallel_for<class k3>(nd_range<3>(gws3, lws3), [=] (nd_item<3> item) {
      kernel_256_winograd_AtIA(item, 
        sm_bias.get_pointer(), sm_scale.get_pointer(), sm_input.get_pointer(),
        i.get_pointer(), b.get_pointer(), s.get_pointer(), o.get_pointer());
    });
  });

  q.submit([&] (handler &cgh) {
    auto acc = output.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();

  nT2 = getTimeMicroseconds64();

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
  free(input_);

  return ((nT2-nT1) << 16);
}
