/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \file
 * Operations for writing linear segments of data from the CUDA thread block
 */

#pragma once

#include <iterator>
#include <type_traits>
#include "utils.hpp"
#include "block_exchange.hpp"

template <
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorT>
inline void StoreDirectWarpStriped(
    int                 linear_tid,                 ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
    T                   (&items)[ITEMS_PER_THREAD]) ///< [out] Data to load
{
    int tid         = linear_tid & (PTX_WARP_THREADS - 1);
    int wid         = linear_tid >> PTX_LOG_WARP_THREADS;
    int warp_offset = wid * PTX_WARP_THREADS * ITEMS_PER_THREAD;

    OutputIteratorT thread_itr = block_itr + warp_offset + tid;

    // Store directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        thread_itr[(ITEM * PTX_WARP_THREADS)] = items[ITEM];
    }
}


/**
 * \brief Store a warp-striped arrangement of data across the thread block into a linear segment of items, guarded by range
 *
 * \warpstriped
 *
 * \par Usage Considerations
 * The number of threads in the thread block must be a multiple of the architecture's warp size.
 *
 * \tparam T                    <b>[inferred]</b> The data type to store.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam OutputIteratorT      <b>[inferred]</b> The random-access iterator type for output \iterator.
 */
template <
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorT>
inline void StoreDirectWarpStriped(
    int                 linear_tid,                 ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
    T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
    int                 valid_items)                ///< [in] Number of valid items to write
{
    int tid         = linear_tid & (PTX_WARP_THREADS - 1);
    int wid         = linear_tid >> PTX_LOG_WARP_THREADS;
    int warp_offset = wid * PTX_WARP_THREADS * ITEMS_PER_THREAD;

    OutputIteratorT thread_itr = block_itr + warp_offset + tid;

    // Store directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (warp_offset + tid + (ITEM * PTX_WARP_THREADS) < valid_items)
        {
            thread_itr[(ITEM * PTX_WARP_THREADS)] = items[ITEM];
        }
    }
}

template <
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorT>
inline void StoreDirectBlocked(
    int                 linear_tid,
    OutputIteratorT     block_itr,
    T                   (&items)[ITEMS_PER_THREAD])
{
    OutputIteratorT thread_itr = block_itr + (linear_tid * ITEMS_PER_THREAD);

    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        thread_itr[ITEM] = items[ITEM];
    }
}


template <
    typename            T,
    int                 ITEMS_PER_THREAD,
    typename            OutputIteratorT>
inline void StoreDirectBlocked(
    int                 linear_tid,
    OutputIteratorT     block_itr,
    T                   (&items)[ITEMS_PER_THREAD],
    int                 valid_items)
{
    OutputIteratorT thread_itr = block_itr + (linear_tid * ITEMS_PER_THREAD);

    // Store directly in thread-blocked order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (ITEM + (linear_tid * ITEMS_PER_THREAD) < valid_items)
        {
            thread_itr[ITEM] = items[ITEM];
        }
    }
}



//-----------------------------------------------------------------------------
// Generic BlockStore abstraction
//-----------------------------------------------------------------------------

enum BlockStoreAlgorithm
{
    BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
    BLOCK_STORE_WARP_TRANSPOSE,
    BLOCK_STORE_DIRECT,
};


template <
    typename                T,
    int                     BLOCK_DIM_X,
    int                     ITEMS_PER_THREAD,
    BlockStoreAlgorithm     ALGORITHM           = BLOCK_STORE_DIRECT,
    int                     BLOCK_DIM_Y         = 1,
    int                     BLOCK_DIM_Z         = 1>
class BlockStore
{
private:
    /******************************************************************************
     * Constants and typed definitions
     ******************************************************************************/

    /// Constants
    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
    };


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Store helper
    template <BlockStoreAlgorithm _POLICY, int DUMMY>
    struct StoreInternal;

    /**
     * BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, DUMMY>
    {
        enum
        {
            WARP_THREADS = WARP_THREADS(0)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        static_assert((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {
            /// Temporary storage for partially-full block guard
            volatile int valid_items;
        };

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline StoreInternal(
            TempStorage &temp_storage,
            int linear_tid,
            const sycl::nd_item<3> &item)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid),
            item(item)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            BlockExchange(temp_storage, item).BlockedToWarpStriped(items);
            StoreDirectWarpStriped(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT   block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage, item).BlockedToWarpStriped(items);
            if (linear_tid == 0)
                temp_storage.valid_items = valid_items;     // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
            item.barrier(sycl::access::fence_space::local_space);

            StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
        }
    };
    /**
     * BLOCK_STORE_WARP_TRANSPOSE specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_WARP_TRANSPOSE, DUMMY>
    {
        enum
        {
            WARP_THREADS = WARP_THREADS(0)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        static_assert((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<T, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {
            /// Temporary storage for partially-full block guard
            volatile int valid_items;
        };

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline StoreInternal(
            TempStorage &temp_storage,
            int linear_tid,
            const sycl::nd_item<3> &item)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid),
            item(item)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT   block_itr,                    ///< [in] The thread block's base output iterator for storing to
            T                 (&items)[ITEMS_PER_THREAD])   ///< [in] Data to store
        {
            BlockExchange(temp_storage, item).BlockedToWarpStriped(items);
            StoreDirectWarpStriped(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT   block_itr,                    ///< [in] The thread block's base output iterator for storing to
            T                 (&items)[ITEMS_PER_THREAD],   ///< [in] Data to store
            int               valid_items)                  ///< [in] Number of valid items to write
        {
            BlockExchange(temp_storage, item).BlockedToWarpStriped(items);
            if (linear_tid == 0)
                temp_storage.valid_items = valid_items;     // Move through volatile smem as a workaround to prevent RF spilling on subsequent loads
            item.barrier(sycl::access::fence_space::local_space);
            StoreDirectWarpStriped(linear_tid, block_itr, items, temp_storage.valid_items);
        }
    };

    /**
     * BLOCK_STORE_DIRECT specialization of store helper
     */
    template <int DUMMY>
    struct StoreInternal<BLOCK_STORE_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline StoreInternal(
            TempStorage &/*temp_storage*/,
            int linear_tid,
            const sycl::nd_item<3> &item)
        :
            linear_tid(linear_tid),
            item(item)
        {}

        /// Store items into a linear segment of memory
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
        {
            StoreDirectBlocked(linear_tid, block_itr, items);
        }

        /// Store items into a linear segment of memory, guarded by range
        template <typename OutputIteratorT>
        inline void Store(
            OutputIteratorT     block_itr,                  ///< [in] The thread block's base output iterator for storing to
            T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
            int                 valid_items)                ///< [in] Number of valid items to write
        {
            StoreDirectBlocked(linear_tid, block_itr, items, valid_items);
        }
    };

    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal load implementation to use
    typedef StoreInternal<ALGORITHM, 0> InternalStore;


    /// Shared memory storage layout type
    typedef typename InternalStore::TempStorage _TempStorage;


    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    inline _TempStorage &PrivateStorage(_TempStorage &private_storage)
    {
        return private_storage;
    }


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Thread reference to shared storage
    _TempStorage &temp_storage;

    /// Linear thread-id
    int linear_tid;

    const sycl::nd_item<3> &item;

public:


    /// \smemstorage{BlockStore}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    inline BlockStore(_TempStorage &private_storage,
                      const sycl::nd_item<3> &item)
        : temp_storage(PrivateStorage(private_storage)),
          linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item)),
          item(item)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    inline BlockStore(
        TempStorage &temp_storage,             ///< [in] Reference to memory allocation having layout type TempStorage
        const sycl::nd_item<3> &item)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item)),
        item(item)
    {}

    template <typename OutputIteratorT>
    inline void Store(
        OutputIteratorT     block_itr,                  ///< [out] The thread block's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD]) ///< [in] Data to store
    {
        InternalStore(temp_storage, linear_tid, item).Store(block_itr, items);
    }

    template <typename OutputIteratorT>
    inline void Store(
        OutputIteratorT     block_itr,                  ///< [out] The thread block's base output iterator for storing to
        T                   (&items)[ITEMS_PER_THREAD], ///< [in] Data to store
        int                 valid_items)                ///< [in] Number of valid items to write
    {
        InternalStore(temp_storage, linear_tid, item).Store(block_itr, items, valid_items);
    }
};
