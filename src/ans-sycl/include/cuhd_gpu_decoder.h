/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_GPU_DECODER_
#define CUHD_GPU_DECODER_

#include "cuhd_constants.h"
#include "cuhd_codetable.h"
#include "cuhd_util.h"
#include "cuhd_subsequence_sync_point.h"
#include "ans_sycl_definitions.h"

namespace cuhd {
    class CUHDGPUDecoder {
        public:
            static void decode(sycl::queue &q,
                               sycl::buffer<UNIT_TYPE, 1> &d_input_buffer,
                               size_t input_size,
                               sycl::buffer<SYMBOL_TYPE, 1> &d_output_buffer,
                               size_t output_size,
                               sycl::buffer<std::uint32_t, 1> &d_table,
                               sycl::buffer<sycl::uint4, 1> &d_sync_info,
                               sycl::buffer<std::uint32_t, 1> &d_output_sizes,
                               sycl::buffer<std::uint8_t, 1> &d_sequence_synced,
                               std::uint8_t* h_sequence_synced,
                               STATE_TYPE initial_state,
                               std::uint32_t initial_bit,
                               std::uint32_t number_of_states,
                               size_t max_codeword_length,
                               size_t preferred_subsequence_size,
                               size_t threads_per_block); 
    };
}

#endif /* CUHD_GPU_DECODER */

