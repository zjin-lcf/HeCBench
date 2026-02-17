/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
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
 * Operations for reading linear tiles of data into the GPU thread block.
 */

#pragma once

#include <iterator>
#include <type_traits>
#include "utils.hpp"
#include "block_exchange.hpp"

template <
    typename        InputT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
inline void LoadDirectWarpStriped(
    int             linear_tid,
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD]) ///< [out] Data to load
{
    int tid                = linear_tid & (PTX_WARP_THREADS - 1);
    int wid                = linear_tid >> PTX_LOG_WARP_THREADS;
    int warp_offset        = wid * PTX_WARP_THREADS * ITEMS_PER_THREAD;

    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        //new(&items[ITEM]) InputT(block_itr[warp_offset + tid + (ITEM * WARP_THREADS)]);
        items[ITEM] = block_itr[warp_offset + tid + (ITEM * PTX_WARP_THREADS)];
    }
}


/**
 * \brief Load a linear segment of items into a warp-striped arrangement across the thread block, guarded by range
 *
 * \warpstriped
 *
 * \par Usage Considerations
 * The number of threads in the thread block must be a multiple of the architecture's warp size.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorT       <b>[inferred]</b> The random-access iterator type for input \iterator.
 */
template <
    typename        InputT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
inline void LoadDirectWarpStriped(
    int             linear_tid,                 ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
    int             valid_items)                ///< [in] Number of valid items to load
{
    int tid                = linear_tid & (PTX_WARP_THREADS - 1);
    int wid                = linear_tid >> PTX_LOG_WARP_THREADS;
    int warp_offset        = wid * PTX_WARP_THREADS * ITEMS_PER_THREAD;

    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if (warp_offset + tid + (ITEM * PTX_WARP_THREADS) < valid_items)
        {
            // new(&items[ITEM]) InputT(block_itr[warp_offset + tid + (ITEM * WARP_THREADS)]);
            items[ITEM] = block_itr[warp_offset + tid + (ITEM * PTX_WARP_THREADS)];
        }
    }
}


/**
 * \brief Load a linear segment of items into a warp-striped arrangement across the thread block, guarded by range, with a fall-back assignment of out-of-bound elements.
 *
 * \warpstriped
 *
 * \par Usage Considerations
 * The number of threads in the thread block must be a multiple of the architecture's warp size.
 *
 * \tparam T                    <b>[inferred]</b> The data type to load.
 * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
 * \tparam InputIteratorT       <b>[inferred]</b> The random-access iterator type for input \iterator.
 */
template <
    typename        InputT,
    typename        DefaultT,
    int             ITEMS_PER_THREAD,
    typename        InputIteratorT>
inline void LoadDirectWarpStriped(
    int             linear_tid,                 ///< [in] A suitable 1D thread-identifier for the calling thread (e.g., <tt>(threadIdx.y * blockDim.x) + linear_tid</tt> for 2D thread blocks)
    InputIteratorT  block_itr,                  ///< [in] The thread block's base input iterator for loading from
    InputT          (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
    int             valid_items,                ///< [in] Number of valid items to load
    DefaultT        oob_default)                ///< [in] Default value to assign out-of-bound items
{
    // Load directly in warp-striped order
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        items[ITEM] = oob_default;

    LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
}



template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
inline void LoadDirectBlocked(int linear_tid, InputIteratorT block_itr,
                                       InputT (&items)[ITEMS_PER_THREAD])
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
    }
}

template <typename InputT, int ITEMS_PER_THREAD, typename InputIteratorT>
inline void
LoadDirectBlocked(int linear_tid, InputIteratorT block_itr,
                  InputT (&items)[ITEMS_PER_THREAD],
                  int valid_items) ///< [in] Number of valid items to load
{

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
    {
        if ((linear_tid * ITEMS_PER_THREAD) + ITEM < valid_items)
        {
            items[ITEM] = block_itr[(linear_tid * ITEMS_PER_THREAD) + ITEM];
        }
    }
}

template <typename InputT, typename DefaultT, int ITEMS_PER_THREAD,
          typename InputIteratorT>
inline void LoadDirectBlocked(int linear_tid, InputIteratorT block_itr,
                                       InputT (&items)[ITEMS_PER_THREAD],
                                       int valid_items, DefaultT oob_default)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        items[ITEM] = oob_default;

    LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
}


//-----------------------------------------------------------------------------
// Generic BlockLoad abstraction
//-----------------------------------------------------------------------------

enum BlockLoadAlgorithm
{
    BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
    BLOCK_LOAD_WARP_TRANSPOSE,
    BLOCK_LOAD_DIRECT
};


template <
    typename            InputT,
    int                 BLOCK_DIM_X,
    int                 ITEMS_PER_THREAD,
    BlockLoadAlgorithm  ALGORITHM           = BLOCK_LOAD_DIRECT,
    int                 BLOCK_DIM_Y         = 1,
    int                 BLOCK_DIM_Z         = 1>
class BlockLoad
{
private:

    /******************************************************************************
     * Constants and typed definitions
     ******************************************************************************/

    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
    };


    /******************************************************************************
     * Algorithmic variants
     ******************************************************************************/

    /// Load helper
    template <BlockLoadAlgorithm _POLICY, int DUMMY>
    struct LoadInternal;

    /**
     * BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, DUMMY>
    {
        enum
        {
            WARP_THREADS = WARP_THREADS(0)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        static_assert((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<InputT, BLOCK_DIM_X, ITEMS_PER_THREAD, true, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {};

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline LoadInternal(
            TempStorage &temp_storage,
            int linear_tid,
            const sycl::nd_item<3> &item)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid),
            item(item)
        {}

        /// Load a linear segment of items from memory
        template <typename InputIteratorT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }

        /// Load a linear segment of items from memory, guarded by range
        template <typename InputIteratorT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
            int             valid_items)                    ///< [in] Number of valid items to load
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }


        /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
        template <typename InputIteratorT, typename DefaultT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
            int             valid_items,                    ///< [in] Number of valid items to load
            DefaultT        oob_default)                    ///< [in] Default value to assign out-of-bound items
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items, oob_default);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }
    };


    /**
     * BLOCK_LOAD_WARP_TRANSPOSE specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_WARP_TRANSPOSE, DUMMY>
    {
        enum
        {
            WARP_THREADS = WARP_THREADS(0)
        };

        // Assert BLOCK_THREADS must be a multiple of WARP_THREADS
        static_assert((int(BLOCK_THREADS) % int(WARP_THREADS) == 0), "BLOCK_THREADS must be a multiple of WARP_THREADS");

        // BlockExchange utility type for keys
        typedef BlockExchange<InputT, BLOCK_DIM_X, ITEMS_PER_THREAD, false, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockExchange;

        /// Shared memory storage layout type
        struct _TempStorage : BlockExchange::TempStorage
        {};

        /// Alias wrapper allowing storage to be unioned
        struct TempStorage : Uninitialized<_TempStorage> {};

        /// Thread reference to shared storage
        _TempStorage &temp_storage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline LoadInternal(
            TempStorage &temp_storage,
            int linear_tid,
            const sycl::nd_item<3> &item)
        :
            temp_storage(temp_storage.Alias()),
            linear_tid(linear_tid),
            item(item)
        {}

        /// Load a linear segment of items from memory
        template <typename InputIteratorT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD])     ///< [out] Data to load{
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }

        /// Load a linear segment of items from memory, guarded by range
        template <typename InputIteratorT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
            int             valid_items)                    ///< [in] Number of valid items to load
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }


        /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
        template <typename InputIteratorT, typename DefaultT>
        inline void Load(
            InputIteratorT  block_itr,                      ///< [in] The thread block's base input iterator for loading from
            InputT          (&items)[ITEMS_PER_THREAD],     ///< [out] Data to load
            int             valid_items,                    ///< [in] Number of valid items to load
            DefaultT        oob_default)                    ///< [in] Default value to assign out-of-bound items
        {
            LoadDirectWarpStriped(linear_tid, block_itr, items, valid_items, oob_default);
            BlockExchange(temp_storage, item).WarpStripedToBlocked(items, items);
        }
    };

    /**
     * BLOCK_LOAD_DIRECT specialization of load helper
     */
    template <int DUMMY>
    struct LoadInternal<BLOCK_LOAD_DIRECT, DUMMY>
    {
        /// Shared memory storage layout type
        typedef NullType TempStorage;

        /// Linear thread-id
        int linear_tid;

        const sycl::nd_item<3> &item;

        /// Constructor
        inline LoadInternal(TempStorage & /*temp_storage*/,
                            int linear_tid,
                            const sycl::nd_item<3> &item) // unused
            : linear_tid(linear_tid), item(item)
        {}

        /// Load a linear segment of items from memory
        template <typename InputIteratorT>
        inline void Load(InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_THREAD])
        {
            LoadDirectBlocked(linear_tid, block_itr, items);
        }

        /// Load a linear segment of items from memory, guarded by range
        template <typename InputIteratorT>
        inline void Load(InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_THREAD],
                                  int valid_items)
        {
            LoadDirectBlocked(linear_tid, block_itr, items, valid_items);
        }

        /// Load a linear segment of items from memory, guarded by range, with a fall-back assignment of out-of-bound elements
        template <typename InputIteratorT, typename DefaultT>
        inline void Load(InputIteratorT block_itr,
                                  InputT (&items)[ITEMS_PER_THREAD],
                                  int valid_items, DefaultT oob_default)
        {
            LoadDirectBlocked(linear_tid, block_itr, items, valid_items, oob_default);
        }

    };


    /******************************************************************************
     * Type definitions
     ******************************************************************************/

    /// Internal load implementation to use
    typedef LoadInternal<ALGORITHM, 0> InternalLoad;


    /// Shared memory storage layout type
    typedef typename InternalLoad::TempStorage _TempStorage;


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

    /// SYCL workitem
    const sycl::nd_item<3> &item;

public:

    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    inline BlockLoad(_TempStorage &private_storage,
                     const sycl::nd_item<3> &item)
        : temp_storage(PrivateStorage(private_storage)),
          linear_tid(
              RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item)),
          item(item)
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    inline BlockLoad(TempStorage &temp_storage,
                             ///< [in] Reference to memory allocation having
                             ///< layout type TempStorage
                     const sycl::nd_item<3> &item) 
        : temp_storage(temp_storage.Alias()),
          linear_tid(
              RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item)),
          item(item)
    {}




    /******************************************************************//**
     * \name Data movement
     *********************************************************************/

    template <typename InputIteratorT>
    inline void
    Load(InputIteratorT block_itr, ///< [in] The thread block's base input
                                   ///< iterator for loading from
         InputT (&items)[ITEMS_PER_THREAD]) ///< [out] Data to load
    {
        InternalLoad(temp_storage, linear_tid, item).Load(block_itr, items);
    }

    template <typename InputIteratorT>
    inline void
    Load(InputIteratorT block_itr, ///< [in] The thread block's base input
                                   ///< iterator for loading from
         InputT (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
         int valid_items) ///< [in] Number of valid items to load
    {
        InternalLoad(temp_storage, linear_tid, item).Load(block_itr, items, valid_items);
    }

    template <typename InputIteratorT, typename DefaultT>
    inline void
    Load(InputIteratorT block_itr, ///< [in] The thread block's base input
                                   ///< iterator for loading from
         InputT (&items)[ITEMS_PER_THREAD], ///< [out] Data to load
         int valid_items, ///< [in] Number of valid items to load
         DefaultT
             oob_default) ///< [in] Default value to assign out-of-bound items
    {
        InternalLoad(temp_storage, linear_tid, item).Load(block_itr, items, valid_items, oob_default);
    }

};

