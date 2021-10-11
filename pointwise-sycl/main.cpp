/* Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"

typedef struct {
  double i, c, h;
} checksum;

// Device functions
inline float sigmoidf(float in) {
  return 1.f / (1.f + sycl::exp(-in));  
}

// Fused kernel
void elementWise_fp(
    nd_item<1> &item,
    int hiddenSize, int miniBatch,
    const float *__restrict tmp_h, 
    const float *__restrict tmp_i, 
    const float *__restrict bias,
    float *__restrict linearGates,
    float *__restrict h_out,
    float *__restrict i_out,
    const float *__restrict c_in,
    float *__restrict c_out)
{
  int index = item.get_global_id(0);
  int numElements = miniBatch * hiddenSize;

  if (index >= numElements) return;

  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;   

  float g[4];

  for (int i = 0; i < 4; i++) {
    g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
    g[i] += bias[i * hiddenSize + index % hiddenSize] + bias[(i + 4) * hiddenSize + index % hiddenSize];
    linearGates[gateIndex + i * hiddenSize] = g[i];
  }   

  float in_gate     = sigmoidf(g[0]);
  float forget_gate = sigmoidf(g[1]);
  float in_gate2    = sycl::tanh(g[2]);
  float out_gate    = sigmoidf(g[3]);

  float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);

  c_out[index] = val;

  val = out_gate * sycl::tanh(val);                                   

  h_out[index] = val;
  i_out[index] = val;
}

float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

void init (nd_item<1> &item, float* data, int size) {
  int index = item.get_global_id(0);
  if (index >= size) return;
  unsigned int seed = index ^ size;
  data[index] = LCG_random(&seed);
}

void test(queue &q, int hiddenSize, int miniBatch, int seqLength, int numLayers, checksum &cs) {

  // Input/output data
  int numElements = hiddenSize * miniBatch;

  int hc_size = (seqLength + 1) * (numLayers) * numElements;
  int i_size = (seqLength) * (numLayers + 1) * numElements;
  int bias_size = numLayers * hiddenSize * 8;
  int tmp_h_size = 4 * numLayers * numElements;
  int tmp_i_size = 4 * seqLength * numElements;

  buffer<float, 1> d_h_data (hc_size);
  buffer<float, 1> d_i_data (i_size);
  buffer<float, 1> d_c_data (hc_size);
  buffer<float, 1> d_bias (bias_size);

  // Workspace
  buffer<float, 1> d_tmp_h (tmp_h_size);
  buffer<float, 1> d_tmp_i (tmp_i_size);

  // Activations
  buffer<float, 1> d_linearGates (4 * seqLength * numLayers * numElements);  

  // Initialise with random values on a device
  range<1> lws (256);
  range<1> gws_hc ((hc_size + 255)/256*256);
  range<1> gws_b ((bias_size + 255)/256*256);
  range<1> gws_tmp_h ((tmp_h_size + 255)/256*256);
  range<1> gws_tmp_i ((tmp_i_size + 255)/256*256);

  q.submit([&] (handler &cgh) {
    auto tmp_h = d_tmp_h.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class init_h_data>(nd_range<1>(gws_tmp_h, lws), [=] (nd_item<1> item) {
      init(item, tmp_h.get_pointer(), tmp_h_size);
    });
  });
  
  q.submit([&] (handler &cgh) {
    auto c_data = d_c_data.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class init_c_data>(nd_range<1>(gws_hc, lws), [=] (nd_item<1> item) {
      init(item, c_data.get_pointer(), hc_size);
    });
  });
  
  q.submit([&] (handler &cgh) {
    auto tmp_i = d_tmp_i.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class init_i_data>(nd_range<1>(gws_tmp_i, lws), [=] (nd_item<1> item) {
      init(item, tmp_i.get_pointer(), tmp_i_size);
    });
  });
  
  q.submit([&] (handler &cgh) {
    auto bias = d_bias.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class init_bias_data>(nd_range<1>(gws_b, lws), [=] (nd_item<1> item) {
      init(item, bias.get_pointer(), bias_size);
    });
  });

  q.wait();

  int lStart = 0;
  int lEnd = 0;
  int rStart = 0;
  int rEnd = 0;
  int recurBatchSize = 2;

  range<1> gws_p ((numElements + 255)/256*256);
  
  while (true) {
    // Many layer "scheduling".
    if (lEnd == 0) {
      lStart = 0;
      lEnd = 1;
      rStart = 0;
    }
    else {
      // Move "up" and "left"
      lStart++;
      lEnd++;

      rStart -= recurBatchSize;

      // Over the top or off the left, reset to layer 0
      if (lEnd > numLayers || rStart < 0) {
        rStart += (lStart + 1) * recurBatchSize;

        lStart = 0;
        lEnd = 1;
      }

      // Off the right, step up
      while (rStart >= seqLength && lEnd <= numLayers) {
        lStart++;
        lEnd++;
        rStart -= recurBatchSize;
      }

      // Over the top or off the left, done!
      if (lEnd > numLayers || rStart < 0) {
        break;
      }
    }

    rEnd = rStart + recurBatchSize;
    if (rEnd > seqLength) rEnd = seqLength;

    for (int layer = lStart; layer < lEnd; layer++) {         
      for (int i = rStart; i < rEnd; i++)
        q.submit([&] (handler &cgh) {
          auto tmp_h = d_tmp_h.get_access<sycl_read>(cgh);
          auto tmp_i = d_tmp_i.get_access<sycl_read>(cgh);
          auto bias = d_bias.get_access<sycl_read>(cgh);
          auto linearGates = d_linearGates.get_access<sycl_write>(cgh);
          auto h_data = d_h_data.get_access<sycl_write>(cgh);
          auto i_data = d_i_data.get_access<sycl_write>(cgh);
          auto c_data = d_c_data.get_access<sycl_read_write>(cgh);
          cgh.parallel_for<class pw>(nd_range<1>(gws_p, lws), [=] (nd_item<1> item) {
            elementWise_fp 
            (item,
	     hiddenSize, miniBatch,
             tmp_h.get_pointer() + 4 * layer * numElements, 
             tmp_i.get_pointer() + 4 * i * numElements, 
             bias.get_pointer() + 8 * layer * hiddenSize,
             linearGates.get_pointer() + 4 * (i * numElements + layer * seqLength * numElements),
             h_data.get_pointer() + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
             i_data.get_pointer() + i * numElements + (layer + 1) * seqLength * numElements,
             c_data.get_pointer() + i * numElements + layer * (seqLength + 1) * numElements,
             c_data.get_pointer() + (i + 1) * numElements + layer * (seqLength + 1) * numElements);
	  });
        });
    }
  }

  float *testOutputi = (float*)malloc(numElements * seqLength * sizeof(float));
  float *testOutputh = (float*)malloc(numElements * numLayers * sizeof(float));
  float *testOutputc = (float*)malloc(numElements * numLayers * sizeof(float));

  q.wait();

  q.submit([&] (handler &cgh) {
    auto acc = d_i_data.get_access<sycl_read>(cgh, 
      range<1>(seqLength * numElements), id<1>(numLayers * seqLength * numElements));
    cgh.copy(acc, testOutputi);
  });

  for (int layer = 0; layer < numLayers; layer++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_h_data.get_access<sycl_read>(cgh,
        range<1>(numElements), id<1>(seqLength * numElements + layer * (seqLength + 1) * numElements));
      cgh.copy(acc, testOutputh + layer * numElements);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_c_data.get_access<sycl_read>(cgh,
        range<1>(numElements), id<1>(seqLength * numElements + layer * (seqLength + 1) * numElements));
      cgh.copy(acc, testOutputc + layer * numElements);
    });
  }
  q.wait();

  double checksumi = 0.;
  double checksumh = 0.;
  double checksumc = 0.;

  for (int m = 0; m < miniBatch; m++) {
    for (int j = 0; j < seqLength; j++) {
      for (int i = 0; i < hiddenSize; i++) {
        checksumi += testOutputi[j * numElements + m * hiddenSize + i];
        //if (hiddenSize <= 8) printf("i: (%d,%d): %E\n", j, i, testOutputi[j * numElements + m * hiddenSize + i]);
      }
    }
    for (int j = 0; j < numLayers; j++) {
      for (int i = 0; i < hiddenSize; i++) {         
        checksumh += testOutputh[j * numElements + m * hiddenSize + i];
        checksumc += testOutputc[j * numElements + m * hiddenSize + i];
      }
    }
  }

  cs.i = checksumi;
  cs.c = checksumc;
  cs.h = checksumh;

  free(testOutputi);
  free(testOutputc);
  free(testOutputh);
}

int main(int argc, char* argv[]) {
  int seqLength;
  int numLayers;
  int hiddenSize;
  int miniBatch; 

  if (argc == 5) {
    seqLength = atoi(argv[1]);
    numLayers = atoi(argv[2]);
    hiddenSize = atoi(argv[3]);
    miniBatch = atoi(argv[4]);   
  }
  else if (argc == 1) {
    printf("Running with default settings\n");
    seqLength = 100;
    numLayers = 4;
    hiddenSize = 512;
    miniBatch = 64;
  }
  else {
    printf("Usage: ./%s <seqLength> <numLayers> <hiddenSize> <miniBatch>\n", argv[1]);
    return 1;      
  }

  printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n",
         seqLength, numLayers, hiddenSize, miniBatch);  

  int numRuns = 100;
  checksum cs;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  for (int run = 0; run < numRuns; run++) {
    test(q, hiddenSize, miniBatch, seqLength, numLayers, cs);
  }

  printf("i checksum %E     ", cs.i);
  printf("c checksum %E     ", cs.c);
  printf("h checksum %E\n", cs.h);
  return 0;
}

