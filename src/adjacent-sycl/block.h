/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include "utils.h"

template <typename T,
          int BLOCK_DIM_X,
          int BLOCK_DIM_Y     = 1,
          int BLOCK_DIM_Z     = 1>
class BlockAdjacentDifference
{
private:

    /***************************************************************************
     * Constants and type definitions
     **************************************************************************/

    /// Constants

    /// The thread block size in threads
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    /// Shared memory storage layout type (last element from each thread's input)
    struct _TempStorage
    {
        T first_items[BLOCK_THREADS];
        T last_items[BLOCK_THREADS];
    };

    /***************************************************************************
     * Thread fields
     **************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    sycl::nd_item<3> item;

public:

    struct TempStorage : Uninitialized<_TempStorage> {};

    inline BlockAdjacentDifference(TempStorage &temp_storage,
                                   const sycl::nd_item<3> &item) 
        : temp_storage(temp_storage.Alias())
        , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item))
        , item(item)
    {}

    template <int ITEMS_PER_THREAD,
              typename OutputType,
              typename DifferenceOpT>
    inline void
    SubtractLeft(T (&input)[ITEMS_PER_THREAD],
                 OutputType (&output)[ITEMS_PER_THREAD],
                 DifferenceOpT difference_op)
    {
      temp_storage.last_items[linear_tid] = input[ITEMS_PER_THREAD - 1];

      CTA_SYNC();

      #pragma unroll
      for (int item = ITEMS_PER_THREAD - 1; item > 0; item--)
      {
        output[item] = difference_op(input[item], input[item - 1]);
      }

      if (linear_tid == 0)
      {
        output[0] = input[0];
      }
      else
      {
        output[0] = difference_op(input[0],
                                  temp_storage.last_items[linear_tid - 1]);
      }
    }
};
