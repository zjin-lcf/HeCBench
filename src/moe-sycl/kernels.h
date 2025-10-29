#include <float.h>
#include <sycl/sycl.hpp>

template <int TPB>
void moeSoftmax(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float* __restrict__ input,
    const bool* __restrict__ finished,
    float* output,
    const int num_cols)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 0> normalizing_factor (cgh);
    sycl::local_accessor<float, 0> float_max (cgh);
    auto kfn = [=] (sycl::nd_item<3> item) {
      const int thread_row_offset = item.get_group(2) * num_cols;

      float threadData(-FLT_MAX);

      // Don't touch finished rows.
      if ((finished != nullptr) && finished[item.get_group(2)]) {
        return;
      }

      for (int ii = item.get_local_id(2); ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData = sycl::fmax(static_cast<float>(input[idx]), threadData);
      }

      const float maxElem = sycl::reduce_over_group(item.get_group(),
                                                    threadData, sycl::maximum<>());

      if (item.get_local_id(2) == 0) {
        float_max = maxElem;
      }
      item.barrier(sycl::access::fence_space::local_space);

      threadData = 0;

      for (int ii = item.get_local_id(2); ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        threadData += sycl::exp((static_cast<float>(input[idx]) - float_max));
      }

      const auto Z = sycl::reduce_over_group(item.get_group(), threadData, sycl::plus<>());

      if (item.get_local_id(2) == 0) {
        normalizing_factor = 1.f / Z;
      }
      item.barrier(sycl::access::fence_space::local_space);

      for (int ii = item.get_local_id(2); ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        const float val = sycl::exp((static_cast<float>(input[idx]) - float_max)) *
            normalizing_factor;
        output[idx] = val;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

struct key_value_pair { // KeyValuePair
  int key;
  float value; 
};

inline key_value_pair arg_max(const key_value_pair &a, const key_value_pair &b) {
   if (a.value > b.value) {
     return {a.key, a.value};
   } else if (a.value == b.value) {
     return {sycl::min(a.key, b.key), a.value};
   } else {
     return {b.key, b.value};
   }
}

template <int TPB>
void moeTopK(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float* __restrict__ inputs_after_softmax,
    const bool* __restrict__ finished,
    float* __restrict__ output,
    int* __restrict__ indices,
    int* __restrict__ source_rows,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<key_value_pair, 1> smem (sycl::range<1>(TPB), cgh);
    auto kfn = [=] (sycl::nd_item<3> item) {
      key_value_pair thread_kvp;

      const int num_tokens = item.get_group_range(2); // number of tokens
      const int token = item.get_group(2);
      const int tid = item.get_local_id(2);

      const bool row_is_active = finished ? !finished[token] : true;
      const int thread_read_offset = token * num_experts;
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f;  // This is OK because inputs are probabilities

        key_value_pair inp_kvp;
        for (int expert = tid; expert < num_experts; expert += TPB) {
          const int idx = thread_read_offset + expert;
          inp_kvp.key = expert;
          inp_kvp.value = inputs_after_softmax[idx];

          for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
            const int prior_winning_expert = indices[k * token + prior_k];

            if (prior_winning_expert == expert) {
              inp_kvp = thread_kvp;
            }
          }
          thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        smem[tid] = thread_kvp;
        item.barrier(sycl::access::fence_space::local_space);

        for (int stride = TPB / 2; stride > 0; stride /= 2) {
          if (tid < stride) {
              smem[tid] = arg_max(smem[tid], smem[tid + stride]);
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (tid == 0) {
          // Ignore experts the node isn't responsible for with expert parallelism
          auto result_kvp = smem[0];
          const int expert = result_kvp.key;
          const bool node_uses_expert = expert >= start_expert && expert < end_expert;
          const bool should_process_row = row_is_active && node_uses_expert;

          const int idx = k * token + k_idx;
          output[idx] = result_kvp.value;
          indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
          assert(indices[idx] >= 0);
          source_rows[idx] = k_idx * num_tokens + token;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

