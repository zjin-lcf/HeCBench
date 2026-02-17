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
 * The cub::BlockScan class provides [<em>collective</em>](index.html#sec0) methods for computing a parallel prefix sum/scan of items partitioned across a CUDA thread block.
 */

#pragma once

#define CTA_SYNC() item.barrier(sycl::access::fence_space::local_space)

#include <sycl/sycl.hpp>
#include "utils.hpp"
#include "block_scan_raking.hpp"
#include "block_scan_warp_scans.hpp"

/******************************************************************************
 * Algorithmic variants
 ******************************************************************************/

/**
 * \brief BlockScanAlgorithm enumerates alternative algorithms for cub::BlockScan to compute a parallel prefix scan across a CUDA thread block.
 */
enum BlockScanAlgorithm
{
    BLOCK_SCAN_RAKING,
    BLOCK_SCAN_RAKING_MEMOIZE,
    BLOCK_SCAN_WARP_SCANS,
};


/******************************************************************************
 * Block scan
 ******************************************************************************/

template <
    typename            T,
    int                 BLOCK_DIM_X,
    BlockScanAlgorithm  ALGORITHM       = BLOCK_SCAN_RAKING,
    int                 BLOCK_DIM_Y     = 1,
    int                 BLOCK_DIM_Z     = 1>
class BlockScan
{
private:

    /******************************************************************************
     * Constants and type definitions
     ******************************************************************************/

    /// Constants
    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
    };

    /**
     * Ensure the template parameterization meets the requirements of the
     * specified algorithm. Currently, the BLOCK_SCAN_WARP_SCANS policy
     * cannot be used with thread block sizes not a multiple of the
     * architectural warp size.
     */
    static const BlockScanAlgorithm SAFE_ALGORITHM =
        ((ALGORITHM == BLOCK_SCAN_WARP_SCANS) && (BLOCK_THREADS % WARP_THREADS(0) != 0)) ?
            BLOCK_SCAN_RAKING :
            ALGORITHM;

    typedef BlockScanWarpScans<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z> WarpScans;
    typedef BlockScanRaking<T, BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, (SAFE_ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)> Raking;

    /// Define the delegate type for the desired algorithm
    using InternalBlockScan =
      conditional_t<SAFE_ALGORITHM == BLOCK_SCAN_WARP_SCANS, WarpScans, Raking>;

    /// Shared memory storage layout type for BlockScan
    typedef typename InternalBlockScan::TempStorage _TempStorage;


    /******************************************************************************
     * Thread fields
     ******************************************************************************/

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;

    const sycl::nd_item<3> &item;
    /******************************************************************************
     * Utility methods
     ******************************************************************************/

    /// Internal storage allocator
    inline _TempStorage &PrivateStorage(_TempStorage &private_storage)
    {

        return private_storage;
    }


    /******************************************************************************
     * Public types
     ******************************************************************************/
public:

    /// \smemstorage{BlockScan}
    struct TempStorage : Uninitialized<_TempStorage> {};


    /******************************************************************//**
     * \name Collective constructors
     *********************************************************************/
    //@{

    /**
     * \brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    inline BlockScan(_TempStorage &private_storage,
                     const sycl::nd_item<3> &item)
        : temp_storage(PrivateStorage(private_storage)),
          item(item),
          linear_tid(
              RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item))
    {}


    /**
     * \brief Collective constructor using the specified memory allocation as temporary storage.
     */
    inline
    BlockScan(TempStorage &temp_storage,
              const sycl::nd_item<3> &item) ///< [in] Reference to memory allocation having
                                            ///< layout type TempStorage
        : temp_storage(temp_storage.Alias()),
          linear_tid(
              RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, item)),
          item(item)
    {}



    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix sum operations
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The value of 0 is applied as the initial value, and is assigned to \p output in <em>thread</em><sub>0</sub>.
     *
     * \par
     * - \identityzero
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix sum of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix sum
     *     BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>1, 1, ..., 1</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>0, 1, ..., 127</tt>.
     *
     */
    inline void ExclusiveSum(
        T input, ///< [in] Calling thread's input item
        T &output) ///< [out] Calling thread's output
                   ///< item (may be aliased to \p input)
    {
        T initial_value{};

        ExclusiveScan(input, output, initial_value, Sum());
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  The value of 0 is applied as the initial value, and is assigned to \p output in <em>thread</em><sub>0</sub>.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \identityzero
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix sum of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix sum
     *     int block_aggregate;
     *     BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>1, 1, ..., 1</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>0, 1, ..., 127</tt>.
     * Furthermore the value \p 128 will be stored in \p block_aggregate for all threads.
     *
     */
    inline void ExclusiveSum(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        T initial_value{};

        ExclusiveScan(input, output, initial_value, Sum(), block_aggregate);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the block-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \identityzero
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an exclusive prefix sum over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total += block_aggregate;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data = d_data[block_offset];
     *
     *         // Collectively compute the block-wide exclusive prefix sum
     *         BlockScan(temp_storage).ExclusiveSum(
     *             thread_data, thread_data, prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         d_data[block_offset] = thread_data;
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>1, 1, 1, 1, 1, 1, 1, 1, ...</tt>.
     * The corresponding output for the first segment will be <tt>0, 1, ..., 127</tt>.
     * The output for the second segment will be <tt>128, 129, ..., 255</tt>.
     *
     * \tparam BlockPrefixCallbackOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixCallbackOp>
    inline void ExclusiveSum(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        ExclusiveScan(input, output, Sum(), block_prefix_callback_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix sum operations (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The value of 0 is applied as the initial value, and is assigned to \p output[0] in <em>thread</em><sub>0</sub>.
     *
     * \par
     * - \identityzero
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix sum of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix sum
     *     BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    inline void ExclusiveSum(
        T (&input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD]
    ) ///< [out] Calling thread's output items (may be aliased
      ///< to \p input)
    {
        T initial_value{};

        ExclusiveScan(input, output, initial_value, Sum());
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  The value of 0 is applied as the initial value, and is assigned to \p output[0] in <em>thread</em><sub>0</sub>.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \identityzero
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix sum of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix sum
     *     int block_aggregate;
     *     BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>{ [0,1,2,3], [4,5,6,7], ..., [508,509,510,511] }</tt>.
     * Furthermore the value \p 512 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    inline void ExclusiveSum(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T initial_value{};

        ExclusiveScan(input, output, initial_value, Sum(), block_aggregate);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the block-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \identityzero
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an exclusive prefix sum over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 512 integer items that are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3)
     * across 128 threads where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total += block_aggregate;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
     *     typedef cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>  BlockStore;
     *     typedef cub::BlockScan<int, 128>                             BlockScan;
     *
     *     // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
     *     __shared__ union {
     *         typename BlockLoad::TempStorage     load;
     *         typename BlockScan::TempStorage     scan;
     *         typename BlockStore::TempStorage    store;
     *     } temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data[4];
     *         BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *
     *         // Collectively compute the block-wide exclusive prefix sum
     *         int block_aggregate;
     *         BlockScan(temp_storage.scan).ExclusiveSum(
     *             thread_data, thread_data, prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>1, 1, 1, 1, 1, 1, 1, 1, ...</tt>.
     * The corresponding output for the first segment will be <tt>0, 1, 2, 3, ..., 510, 511</tt>.
     * The output for the second segment will be <tt>512, 513, 514, 515, ..., 1022, 1023</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixCallbackOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename BlockPrefixCallbackOp>
    inline void ExclusiveSum(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        ExclusiveScan(input, output, Sum(), block_prefix_callback_op);
    }



    //@}  end member group        // Exclusive prefix sums
    /******************************************************************//**
     * \name Exclusive prefix scan operations
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix max scan of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix max scan
     *     BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>0, -1, 2, -3, ..., 126, -127</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>INT_MIN, 0, 0, 2, ..., 124, 126</tt>.
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void ExclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        T initial_value, ///< [in] Initial value to seed the exclusive scan (and
                         ///< is assigned to \p output[0] in
                         ///< <em>thread</em><sub>0</sub>)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        InternalBlockScan(temp_storage, item)
            .ExclusiveScan(input, output, initial_value, scan_op);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix max scan of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix max scan
     *     int block_aggregate;
     *     BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, Max(), block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>0, -1, 2, -3, ..., 126, -127</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>INT_MIN, 0, 0, 2, ..., 124, 126</tt>.
     * Furthermore the value \p 126 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void ExclusiveScan(
        T input,   ///< [in] Calling thread's input items
        T &output, ///< [out] Calling thread's output items (may be aliased to
                   ///< \p input)
        T initial_value, ///< [in] Initial value to seed the exclusive scan (and
                         ///< is assigned to \p output[0] in
                         ///< <em>thread</em><sub>0</sub>)
        ScanOp scan_op,  ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        InternalBlockScan(temp_storage, item)
            .ExclusiveScan(input, output, initial_value, scan_op,
                           block_aggregate);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an exclusive prefix max scan over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(INT_MIN);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data = d_data[block_offset];
     *
     *         // Collectively compute the block-wide exclusive prefix max scan
     *         BlockScan(temp_storage).ExclusiveScan(
     *             thread_data, thread_data, INT_MIN, Max(), prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         d_data[block_offset] = thread_data;
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>0, -1, 2, -3, 4, -5, ...</tt>.
     * The corresponding output for the first segment will be <tt>INT_MIN, 0, 0, 2, ..., 124, 126</tt>.
     * The output for the second segment will be <tt>126, 128, 128, 130, ..., 252, 254</tt>.
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixCallbackOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename ScanOp,
              typename BlockPrefixCallbackOp>
    inline void ExclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op, ///< [in] Binary scan functor
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        InternalBlockScan(temp_storage, item)
            .ExclusiveScan(input, output, scan_op, block_prefix_callback_op);
    }


    //@}  end member group        // Inclusive prefix sums
    /******************************************************************//**
     * \name Exclusive prefix scan operations (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix max scan of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix max scan
     *     BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }</tt>.
     * The corresponding output \p thread_data in those threads will be
     * <tt>{ [INT_MIN,0,0,2], [2,4,4,6], ..., [506,508,508,510] }</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void ExclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        T initial_value, ///< [in] Initial value to seed the exclusive scan (and
                         ///< is assigned to \p output[0] in
                         ///< <em>thread</em><sub>0</sub>)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        // Reduce consecutive thread items in registers
        T thread_prefix = ThreadReduce(input, scan_op);

        // Exclusive thread block-scan
        ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op);

        // Exclusive scan in registers with prefix as seed
        ThreadScanExclusive(input, output, scan_op, thread_prefix);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an exclusive prefix max scan of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide exclusive prefix max scan
     *     int block_aggregate;
     *     BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, Max(), block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>{ [INT_MIN,0,0,2], [2,4,4,6], ..., [506,508,508,510] }</tt>.
     * Furthermore the value \p 510 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void ExclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        T initial_value, ///< [in] Initial value to seed the exclusive scan (and
                         ///< is assigned to \p output[0] in
                         ///< <em>thread</em><sub>0</sub>)
        ScanOp scan_op,  ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_prefix = ThreadReduce(input, scan_op);

        // Exclusive thread block-scan
        ExclusiveScan(thread_prefix, thread_prefix, initial_value, scan_op,
                      block_aggregate);

        // Exclusive scan in registers with prefix as seed
        ThreadScanExclusive(input, output, scan_op, thread_prefix);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an exclusive prefix max scan over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
     *     typedef cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>  BlockStore;
     *     typedef cub::BlockScan<int, 128>                             BlockScan;
     *
     *     // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
     *     __shared__ union {
     *         typename BlockLoad::TempStorage     load;
     *         typename BlockScan::TempStorage     scan;
     *         typename BlockStore::TempStorage    store;
     *     } temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data[4];
     *         BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *
     *         // Collectively compute the block-wide exclusive prefix max scan
     *         BlockScan(temp_storage.scan).ExclusiveScan(
     *             thread_data, thread_data, INT_MIN, Max(), prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>0, -1, 2, -3, 4, -5, ...</tt>.
     * The corresponding output for the first segment will be <tt>INT_MIN, 0, 0, 2, 2, 4, ..., 508, 510</tt>.
     * The output for the second segment will be <tt>510, 512, 512, 514, 514, 516, ..., 1020, 1022</tt>.
     *
     * \tparam ITEMS_PER_THREAD         <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp                   <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixCallbackOp    <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <int ITEMS_PER_THREAD, typename ScanOp,
              typename BlockPrefixCallbackOp>
    inline void ExclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op,                ///< [in] Binary scan functor
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        // Reduce consecutive thread items in registers
        T thread_prefix = ThreadReduce(input, scan_op);

        // Exclusive thread block-scan
        ExclusiveScan(thread_prefix, thread_prefix, scan_op,
                      block_prefix_callback_op);

        // Exclusive scan in registers with prefix as seed
        ThreadScanExclusive(input, output, scan_op, thread_prefix);
    }


    //@}  end member group
#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document no-initial-value scans

    /******************************************************************//**
     * \name Exclusive prefix scan operations (no initial value, single datum per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  With no initial value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void ExclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        InternalBlockScan(temp_storage, item)
            .ExclusiveScan(input, output, scan_op);
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no initial value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void ExclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op, ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        InternalBlockScan(temp_storage, item)
            .ExclusiveScan(input, output, scan_op, block_aggregate);
    }

    //@}  end member group
    /******************************************************************//**
     * \name Exclusive prefix scan operations (no initial value, multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  With no initial value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void ExclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive thread block-scan
        ExclusiveScan(thread_partial, thread_partial, scan_op);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
    }


    /**
     * \brief Computes an exclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.  With no initial value, the output computed for <em>thread</em><sub>0</sub> is undefined.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void ExclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op,                ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        // Reduce consecutive thread items in registers
        T thread_partial = ThreadReduce(input, scan_op);

        // Exclusive thread block-scan
        ExclusiveScan(thread_partial, thread_partial, scan_op, block_aggregate);

        // Exclusive scan in registers with prefix
        ThreadScanExclusive(input, output, scan_op, thread_partial, (linear_tid != 0));
    }


    //@}  end member group
#endif // DOXYGEN_SHOULD_SKIP_THIS  // Do not document no-initial-value scans

    /******************************************************************//**
     * \name Inclusive prefix sum operations
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.
     *
     * \par
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix sum of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix sum
     *     BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>1, 1, ..., 1</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>1, 2, ..., 128</tt>.
     *
     */
    inline void InclusiveSum(
        T input, ///< [in] Calling thread's input item
        T &output) ///< [out] Calling thread's output
                   ///< item (may be aliased to \p input)
    {
        InclusiveScan(input, output, Sum());
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix sum of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix sum
     *     int block_aggregate;
     *     BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>1, 1, ..., 1</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>1, 2, ..., 128</tt>.
     * Furthermore the value \p 128 will be stored in \p block_aggregate for all threads.
     *
     */
    inline void InclusiveSum(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        InclusiveScan(input, output, Sum(), block_aggregate);
    }



    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes one input element.  Instead of using 0 as the block-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an inclusive prefix sum over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total += block_aggregate;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data = d_data[block_offset];
     *
     *         // Collectively compute the block-wide inclusive prefix sum
     *         BlockScan(temp_storage).InclusiveSum(
     *             thread_data, thread_data, prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         d_data[block_offset] = thread_data;
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>1, 1, 1, 1, 1, 1, 1, 1, ...</tt>.
     * The corresponding output for the first segment will be <tt>1, 2, ..., 128</tt>.
     * The output for the second segment will be <tt>129, 130, ..., 256</tt>.
     *
     * \tparam BlockPrefixCallbackOp          <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename BlockPrefixCallbackOp>
    inline void InclusiveSum(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        InclusiveScan(input, output, Sum(), block_prefix_callback_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix sum operations (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix sum of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix sum
     *     BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>{ [1,2,3,4], [5,6,7,8], ..., [509,510,511,512] }</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     */
    template <int ITEMS_PER_THREAD>
    inline void InclusiveSum(
        T (&input)[ITEMS_PER_THREAD], ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD]) ///< [out] Calling thread's output items (may be aliased
                                       ///< to \p input)
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0]);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum scan_op;
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix, thread_prefix);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix sum of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix sum
     *     int block_aggregate;
     *     BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ [1,1,1,1], [1,1,1,1], ..., [1,1,1,1] }</tt>.  The
     * corresponding output \p thread_data in those threads will be
     * <tt>{ [1,2,3,4], [5,6,7,8], ..., [509,510,511,512] }</tt>.
     * Furthermore the value \p 512 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD>
    inline void InclusiveSum(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0], block_aggregate);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum scan_op;
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix, thread_prefix, block_aggregate);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using addition (+) as the scan operator.  Each thread contributes an array of consecutive input elements.  Instead of using 0 as the block-wide prefix, the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an inclusive prefix sum over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 512 integer items that are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3)
     * across 128 threads where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total += block_aggregate;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
     *     typedef cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>  BlockStore;
     *     typedef cub::BlockScan<int, 128>                             BlockScan;
     *
     *     // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
     *     __shared__ union {
     *         typename BlockLoad::TempStorage     load;
     *         typename BlockScan::TempStorage     scan;
     *         typename BlockStore::TempStorage    store;
     *     } temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data[4];
     *         BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *
     *         // Collectively compute the block-wide inclusive prefix sum
     *         BlockScan(temp_storage.scan).IncluisveSum(
     *             thread_data, thread_data, prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>1, 1, 1, 1, 1, 1, 1, 1, ...</tt>.
     * The corresponding output for the first segment will be <tt>1, 2, 3, 4, ..., 511, 512</tt>.
     * The output for the second segment will be <tt>513, 514, 515, 516, ..., 1023, 1024</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam BlockPrefixCallbackOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename BlockPrefixCallbackOp>
    inline void InclusiveSum(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveSum(input[0], output[0], block_prefix_callback_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            Sum scan_op;
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan
            ExclusiveSum(thread_prefix, thread_prefix, block_prefix_callback_op);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, scan_op, thread_prefix);
        }
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scan operations
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix max scan of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix max scan
     *     BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>0, -1, 2, -3, ..., 126, -127</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>0, 0, 2, 2, ..., 126, 126</tt>.
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void InclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        InternalBlockScan(temp_storage, item)
            .InclusiveScan(input, output, scan_op);
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix max scan of 128 integer items that
     * are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain input item for each thread
     *     int thread_data;
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix max scan
     *     int block_aggregate;
     *     BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, Max(), block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>0, -1, 2, -3, ..., 126, -127</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>0, 0, 2, 2, ..., 126, 126</tt>.
     * Furthermore the value \p 126 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ScanOp   <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <typename ScanOp>
    inline void InclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op, ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        InternalBlockScan(temp_storage, item)
            .InclusiveScan(input, output, scan_op, block_aggregate);
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes one input element.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - Supports non-commutative scan operators.
     * - \rowmajor
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an inclusive prefix max scan over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(INT_MIN);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data = d_data[block_offset];
     *
     *         // Collectively compute the block-wide inclusive prefix max scan
     *         BlockScan(temp_storage).InclusiveScan(
     *             thread_data, thread_data, Max(), prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         d_data[block_offset] = thread_data;
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>0, -1, 2, -3, 4, -5, ...</tt>.
     * The corresponding output for the first segment will be <tt>0, 0, 2, 2, ..., 126, 126</tt>.
     * The output for the second segment will be <tt>128, 128, 130, 130, ..., 254, 254</tt>.
     *
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixCallbackOp        <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <typename ScanOp,
              typename BlockPrefixCallbackOp>
    inline void InclusiveScan(
        T input,   ///< [in] Calling thread's input item
        T &output, ///< [out] Calling thread's output item (may be aliased to \p
                   ///< input)
        ScanOp scan_op, ///< [in] Binary scan functor
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        InternalBlockScan(temp_storage, item)
            .InclusiveScan(input, output, scan_op, block_prefix_callback_op);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive prefix scan operations (multiple data per thread)
     *********************************************************************/
    //@{


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix max scan of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix max scan
     *     BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, Max());
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is <tt>{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }</tt>.  The
     * corresponding output \p thread_data in those threads will be <tt>{ [0,0,2,2], [4,4,6,6], ..., [508,508,510,510] }</tt>.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void InclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op) ///< [in] Binary scan functor
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan
            ExclusiveScan(thread_prefix, thread_prefix, scan_op);

            // Inclusive scan in registers with prefix as seed (first thread does not seed)
            ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates an inclusive prefix max scan of 512 integer items that
     * are partitioned in a [<em>blocked arrangement</em>](index.html#sec5sec3) across 128 threads
     * where each thread owns 4 consecutive items.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * __global__ void ExampleKernel(...)
     * {
     *     // Specialize BlockScan for a 1D block of 128 threads of type int
     *     typedef cub::BlockScan<int, 128> BlockScan;
     *
     *     // Allocate shared memory for BlockScan
     *     __shared__ typename BlockScan::TempStorage temp_storage;
     *
     *     // Obtain a segment of consecutive items that are blocked across threads
     *     int thread_data[4];
     *     ...
     *
     *     // Collectively compute the block-wide inclusive prefix max scan
     *     int block_aggregate;
     *     BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, Max(), block_aggregate);
     *
     * \endcode
     * \par
     * Suppose the set of input \p thread_data across the block of threads is
     * <tt>{ [0,-1,2,-3], [4,-5,6,-7], ..., [508,-509,510,-511] }</tt>.
     * The corresponding output \p thread_data in those threads will be
     * <tt>{ [0,0,2,2], [4,4,6,6], ..., [508,508,510,510] }</tt>.
     * Furthermore the value \p 510 will be stored in \p block_aggregate for all threads.
     *
     * \tparam ITEMS_PER_THREAD     <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp               <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <int ITEMS_PER_THREAD,
              typename ScanOp>
    inline void InclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op,                ///< [in] Binary scan functor
        T &block_aggregate) ///< [out] block-wide aggregate reduction of input items
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op, block_aggregate);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan (with no initial value)
            ExclusiveScan(thread_prefix, thread_prefix, scan_op, block_aggregate);

            // Inclusive scan in registers with prefix as seed (first thread does not seed)
            ThreadScanInclusive(input, output, scan_op, thread_prefix, (linear_tid != 0));
        }
    }


    /**
     * \brief Computes an inclusive block-wide prefix scan using the specified binary \p scan_op functor.  Each thread contributes an array of consecutive input elements.  the call-back functor \p block_prefix_callback_op is invoked by the first warp in the block, and the value returned by <em>lane</em><sub>0</sub> in that warp is used as the "seed" value that logically prefixes the thread block's scan inputs.  Also provides every thread with the block-wide \p block_aggregate of all inputs.
     *
     * \par
     * - The \p block_prefix_callback_op functor must implement a member function <tt>T operator()(T block_aggregate)</tt>.
     *   The functor's input parameter \p block_aggregate is the same value also returned by the scan operation.
     *   The functor will be invoked by the first warp of threads in the block, however only the return value from
     *   <em>lane</em><sub>0</sub> is applied as the block-wide prefix.  Can be stateful.
     * - Supports non-commutative scan operators.
     * - \blocked
     * - \granularity
     * - \smemreuse
     *
     * \par Snippet
     * The code snippet below illustrates a single thread block that progressively
     * computes an inclusive prefix max scan over multiple "tiles" of input using a
     * prefix functor to maintain a running total between block-wide scans.  Each tile consists
     * of 128 integer items that are partitioned across 128 threads.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/block/block_scan.cuh>
     *
     * // A stateful callback functor that maintains a running prefix to be applied
     * // during consecutive scan operations.
     * struct BlockPrefixCallbackOp
     * {
     *     // Running prefix
     *     int running_total;
     *
     *     // Constructor
     *     __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}
     *
     *     // Callback operator to be entered by the first warp of threads in the block.
     *     // Thread-0 is responsible for returning a value for seeding the block-wide scan.
     *     __device__ int operator()(int block_aggregate)
     *     {
     *         int old_prefix = running_total;
     *         running_total = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
     *         return old_prefix;
     *     }
     * };
     *
     * __global__ void ExampleKernel(int *d_data, int num_items, ...)
     * {
     *     // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
     *     typedef cub::BlockLoad<int*, 128, 4, BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     *     typedef cub::BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>  BlockStore;
     *     typedef cub::BlockScan<int, 128>                             BlockScan;
     *
     *     // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
     *     __shared__ union {
     *         typename BlockLoad::TempStorage     load;
     *         typename BlockScan::TempStorage     scan;
     *         typename BlockStore::TempStorage    store;
     *     } temp_storage;
     *
     *     // Initialize running total
     *     BlockPrefixCallbackOp prefix_op(0);
     *
     *     // Have the block iterate over segments of items
     *     for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
     *     {
     *         // Load a segment of consecutive items that are blocked across threads
     *         int thread_data[4];
     *         BlockLoad(temp_storage.load).Load(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *
     *         // Collectively compute the block-wide inclusive prefix max scan
     *         BlockScan(temp_storage.scan).InclusiveScan(
     *             thread_data, thread_data, Max(), prefix_op);
     *         CTA_SYNC();
     *
     *         // Store scanned items to output segment
     *         BlockStore(temp_storage.store).Store(d_data + block_offset, thread_data);
     *         CTA_SYNC();
     *     }
     * \endcode
     * \par
     * Suppose the input \p d_data is <tt>0, -1, 2, -3, 4, -5, ...</tt>.
     * The corresponding output for the first segment will be <tt>0, 0, 2, 2, 4, 4, ..., 510, 510</tt>.
     * The output for the second segment will be <tt>512, 512, 514, 514, 516, 516, ..., 1022, 1022</tt>.
     *
     * \tparam ITEMS_PER_THREAD         <b>[inferred]</b> The number of consecutive items partitioned onto each thread.
     * \tparam ScanOp                   <b>[inferred]</b> Binary scan functor  type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam BlockPrefixCallbackOp    <b>[inferred]</b> Call-back functor type having member <tt>T operator()(T block_aggregate)</tt>
     */
    template <int ITEMS_PER_THREAD, typename ScanOp,
              typename BlockPrefixCallbackOp>
    inline void InclusiveScan(
        T (&input)[ITEMS_PER_THREAD],  ///< [in] Calling thread's input items
        T (&output)[ITEMS_PER_THREAD], ///< [out] Calling thread's output items
                                       ///< (may be aliased to \p input)
        ScanOp scan_op,                ///< [in] Binary scan functor
        BlockPrefixCallbackOp &block_prefix_callback_op
    ) ///< [in-out] <b>[<em>warp</em><sub>0</sub> only]</b>
      ///< Call-back functor for specifying a block-wide prefix
      ///< to be applied to the logical input sequence.
    {
        if (ITEMS_PER_THREAD == 1)
        {
            InclusiveScan(input[0], output[0], scan_op, block_prefix_callback_op);
        }
        else
        {
            // Reduce consecutive thread items in registers
            T thread_prefix = ThreadReduce(input, scan_op);

            // Exclusive thread block-scan
            ExclusiveScan(thread_prefix, thread_prefix, scan_op,
                          block_prefix_callback_op);

            // Inclusive scan in registers with prefix as seed
            ThreadScanInclusive(input, output, scan_op, thread_prefix);
        }
    }
};

