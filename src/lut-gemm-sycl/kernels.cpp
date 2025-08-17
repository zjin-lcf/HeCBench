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

#include <sycl/sycl.hpp>
#include "kernels.h"
#include "mv_fp16.h"

void matmul(sycl::queue &q, void* output, nQWeight_fp16 &nqW, void* input, int n, int algo);
void matmul(sycl::queue &q, void* output, void* input, nQWeight_fp16 &nqW, int m, int algo);

/************************** float16 ***********************/

void matmul(sycl::queue &q, void* output, nQWeight_fp16 &nqW, void* input, int n, int algo){
    q.memset(output, 0, sizeof(sycl::half) * nqW.mSize); // 0.007ms 0.04
    if (nqW.q_bias == nullptr)
        nqmv(q, (sycl::half *)output, nqW, (sycl::half *)input, algo);
    else nqmv_bias(q, (sycl::half *)output, nqW, (sycl::half *)input, algo);
}
void matmul(sycl::queue &q, void* output, void* input, nQWeight_fp16 &nqW, int m, int algo){
    q.memset(output, 0, sizeof(sycl::half) * nqW.mSize);
    if (nqW.q_bias == nullptr)
        nqmv(q, (sycl::half *)output, nqW, (sycl::half *)input, algo);
    else nqmv_bias(q, (sycl::half *)output, nqW, (sycl::half *)input, algo);
}
