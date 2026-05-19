#pragma once

#include "pairwise_metrics.h"

#include <ATen/core/grad_mode.h>
#include <torch/library.h>

#include "braycurtis.h"
#include "utils/scalar_type_utils.h"

namespace torchpairwise {
    namespace ops {
        at::Tensor braycurtis_distances_functor::call(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2) {
            C10_LOG_API_USAGE_ONCE("torchpairwise.csrc.ops.pairwise_metrics.braycurtis_distances")
            at::Tensor x1_, x2_;
            std::tie(x1_, x2_) = utils::check_pairwise_inputs("braycurtis_distances", x1, x2);
            return _braycurtis(x1_, x2_);
        }

        TORCH_LIBRARY_FRAGMENT(torchpairwise, m) {
#define REGISTER_FUNCTOR(FUNCTOR) \
    m.def(c10::str("torchpairwise::", FUNCTOR::schema_str).c_str(), TORCH_FN(FUNCTOR::call))
            REGISTER_FUNCTOR(braycurtis_distances_functor);
        }
    }
}
