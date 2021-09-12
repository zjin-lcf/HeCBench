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

#include <cmath>
#include <stdio.h>

inline int compare_output(FP *output, FP *ref, int dim) {
    int i;
    int fail = 0;
    for(i = 0; i < dim; i++) {
        FP diff = fabs(ref[i] - output[i]);
        if((diff - 0.0f) > 0.00001f && diff > 0.01 * fabs(ref[i])) {
            printf("Failed at line: %d ref: %f actual: %f diff: %f\n", i, ref[i], output[i], diff);
            fail = 1;
	    break;
        }
    }
    return fail;
}

// Sequential transposition for comparison purposes
//[w][h/t][t] to [h/t][w][t]
inline void cpu_soa_asta(FP *src, FP *dst, int height, int width, int tile_size) {
    // We only support height == multiple of tile size
    if((height / tile_size) * tile_size == height)
        for(int k = 0; k < width; k++) {
            for(int i = 0; i < height / tile_size; i++) { //For all tiles
                for(int j = 0; j < tile_size; j++) {
                    //from src[k][i][j] to dst[i][k][j]
                    dst[i * width * tile_size + k * tile_size + j] = src[k * height + i * tile_size + j];
                }
            }
        }
}

inline int verify(FP *input2, FP *input, int height, int width, int tile_size) {
    FP *output = (FP *)malloc(width * height * sizeof(FP));
    cpu_soa_asta(input, output, height, width, tile_size);
    int status = compare_output(input2, output, height * width);
    free(output);
    return status;
}
