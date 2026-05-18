#pragma once

#include <ATen/ATen.h>

#include "../macros.h"

namespace torchpairwise {
    namespace ops {
        namespace utils {
            template<bool is_boolean_metric = false, bool is_2d_metric = false>
            static C10_ALWAYS_INLINE std::pair<at::Tensor, at::Tensor> check_pairwise_inputs(
                    const at::CheckedFrom &c,
                    const at::Tensor &x1,
                    const c10::optional<at::Tensor> &x2 = c10::nullopt) {
                bool unbatched = x1.ndimension() == (is_2d_metric ? 3 : 2);
                if constexpr (is_2d_metric) {
                    TORCH_CHECK(unbatched || x1.ndimension() == 4,
                                "x1 must be 3D tensor (unbatched) or 4D tensor (batched)")
                } else {
                    TORCH_CHECK(unbatched || x1.ndimension() == 3,
                                "x1 must be 2D tensor (unbatched) or 3D tensor (batched)")
                }
                if (x2.has_value()) {
                    auto x2_ = x2.value();
                    if constexpr (is_2d_metric) {
                        TORCH_CHECK(unbatched || x2_.ndimension() == 4,
                                    "x2 must be 3D tensor (unbatched) or 4D tensor (batched)")
                        TORCH_CHECK((unbatched && x1.size(2) == x2_.size(2)) ||
                                    (!unbatched && x1.size(3) == x2_.size(3)),
                                    "x1 and x2 must have same number of features. Got: x1.size(",
                                    unbatched ? 2 : 3,
                                    ")=",
                                    x1.size(unbatched ? 2 : 3),
                                    ", x2.size(",
                                    unbatched ? 2 : 3, ")=",
                                    x2_.size(unbatched ? 2 : 3))
                    } else {
                        TORCH_CHECK(unbatched || x2_.ndimension() == 3,
                                    "x2 must be 2D tensor (unbatched) or 3D tensor (batched)")
                        TORCH_CHECK((unbatched && x1.size(1) == x2_.size(1)) ||
                                    (!unbatched && x1.size(2) == x2_.size(2)),
                                    "x1 and x2 must have same number of features. Got: x1.size(",
                                    unbatched ? 1 : 2,
                                    ")=",
                                    x1.size(unbatched ? 1 : 2),
                                    ", x2.size(",
                                    unbatched ? 1 : 2, ")=",
                                    x2_.size(unbatched ? 1 : 2))
                    }
                    if constexpr (is_boolean_metric) {
                        if (x1.scalar_type() != at::kBool || x2_.scalar_type() != at::kBool) {
                            TORCH_WARN("Data was converted to ", at::kBool, " for metric ", c)
                        }
                        return std::make_pair(x1.to(at::kBool), x2_.to(at::kBool));
                    } else
                        return std::make_pair(x1, x2_);
                } else {
                    if constexpr (is_boolean_metric) {
                        if (x1.scalar_type() != at::kBool) {
                            TORCH_WARN("Data was converted to ", at::kBool, " for metric ", c)
                        }
                        auto x1_ = x1.to(at::kBool);
                        return std::make_pair(x1_, x1_);
                    } else
                        return std::make_pair(x1, x1);
                }
            }
        } // namespace utils

        // ~~~~~ functors ~~~~~
        // Note: these functors do not follow torch's native ops convention
        // sklearn
        struct TORCHPAIRWISE_API braycurtis_distances_functor {
            using schema = at::Tensor (const at::Tensor &, const c10::optional<at::Tensor> &);
            using ptr_schema = schema*;
            //static constexpr const char *name = "torchpairwise::braycurtis_distances";
            //static constexpr const char *overload_name = "";
            //static constexpr const char *schema_str = "braycurtis_distances(Tensor x1, Tensor? x2=None) -> Tensor";
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "torchpairwise::braycurtis_distances")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
            STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "braycurtis_distances(Tensor x1, Tensor? x2=None) -> Tensor")
            static at::Tensor call(const at::Tensor &x1, const c10::optional<at::Tensor> &x2);
        };


        inline at::Tensor braycurtis_distances(
                const at::Tensor &x1,
                const c10::optional<at::Tensor> &x2 = c10::nullopt) {
            return braycurtis_distances_functor::call(x1, x2);
        }
    }
}
