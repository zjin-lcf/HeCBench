/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include "common.h"
#include <math.h>

inline void compare_output(T *outp, T *outpCPU, int size) {
    double sum_delta2, sum_ref2, L1norm2;
    sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2    = 0;
    for(int i = 0; i < size; i++) {
        sum_delta2 += fabs(outp[i] - outpCPU[i]);
        sum_ref2 += fabs(outpCPU[i]);
    }
    L1norm2 = (double)(sum_delta2 / sum_ref2);
    if(L1norm2 >= 1e-6){
        printf("Test failed\n");
        exit(1);
    } else {
        printf("Test Passed\n");
    }
}

// Sequential implementation for comparison purposes
inline void cpu_padding(T *matrix, int x_size, int y_size, int pad_size) {
    // start timer
    for(int my_y = y_size - 1; my_y >= 0; --my_y) {
        for(int my_x = pad_size - 1; my_x >= 0; --my_x) {
            if(my_x < x_size)
                matrix[my_y * pad_size + my_x] = matrix[my_y * x_size + my_x];
            else
                matrix[my_y * pad_size + my_x] = 0.0f;
        }
    }
}

inline void verify(T *in_out, T *in_backup, int n, int m, int n_pad) {
    cpu_padding(in_backup, n, m, n_pad);
    compare_output(in_out, in_backup, m * n_pad);
}
