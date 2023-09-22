/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef __HIP_MACRO_H__
#define __HIP_MACRO_H__

#include <hip/hip_runtime.h>

#define CHECK_HIP(call) {                                                   \
    hipError_t err = call;                                                  \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "HIP error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_HIPBLAS(call) {                                                 \
    hipblasStatus_t status = call;                                            \
    if( HIPBLAS_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "HIPBLAS error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_HIPRAND(call) {                                                 \
    hiprandStatus_t status = call;                                            \
    if( HIPRAND_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "HIPRAND error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_ERROR(errorMessage) {                                          \
    hipError_t err = hipGetLastError();                                    \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "HIP error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, hipGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }}
#endif
