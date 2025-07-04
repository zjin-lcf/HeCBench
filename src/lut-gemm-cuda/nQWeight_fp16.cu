/* LUT-GEMM
 * Copyright (c) 2024-present NAVER Cloud Corp. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <cuda.h>

#include "nQWeight_fp16.h"

void nQWeight_fp16::parsing(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups, float* q_bias){
    this->num_groups = num_alpha_groups;
    this->group_size =  kSize/num_alpha_groups;

    __half* p_alpha;
    __half* p_q_bias;
    nb=num_bits;
    this->is_row_wise_quantize = is_row_wise_quantize;
    if(is_row_wise_quantize){
        mSize = row; 
        kSize = col; 
    }
    else{
        mSize = col; 
        kSize = row;             
    }

    if(q_bias == nullptr) p_q_bias = nullptr;
    else{
        cudaMallocManaged(&p_q_bias    ,sizeof(__half  ) * num_groups * mSize);
        for(int i=0;i<num_groups*mSize;i++) p_q_bias[i] = __float2half(q_bias[i]);
    }
    
    cudaMallocManaged(&p_alpha    ,sizeof(__half  ) * num_groups * mSize * nb);
    for(int i=0;i<num_groups*mSize*nb;i++) p_alpha[i] = __float2half(A[i]);

    cudaMallocManaged(&bWeight  ,sizeof(uint32_t) * kSize * mSize * nb / 32);
    cudaMemcpy(bWeight, bW      ,sizeof(uint32_t) * kSize * mSize * nb / 32,    cudaMemcpyHostToDevice);
    this->alpha = (void*)p_alpha;
    this->q_bias = (void*)p_q_bias;
}

nQWeight_fp16::~nQWeight_fp16(){
    cudaFree(alpha);
    cudaFree(bWeight);
    if(q_bias!= nullptr) cudaFree(q_bias);
}
