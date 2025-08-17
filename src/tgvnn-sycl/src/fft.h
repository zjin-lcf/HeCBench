//==---- fft_utils.hpp ----------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef FFT_H__
#define FFT_H__

#include <optional>
#include <utility>
#include <oneapi/mkl.hpp>


namespace fft {

enum class library_data_t : unsigned char {
  complex_float = 0
};

/// An enumeration type to describe the FFT direction is forward or backward.
enum fft_direction : int {
  forward = 0,
  backward
};
/// An enumeration type to describe the types of FFT input and output data.
enum fft_type : int {
  complex_float_to_complex_float = 0
};

/// A class to perform FFT calculation.
class fft_engine {
public:
  /// Default constructor.
  fft_engine() {}
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] input_type Input data type.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] output_type Output data type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, int *n, int *inembed,
              int istride, int idist, library_data_t input_type, int *onembed,
              int ostride, int odist, library_data_t output_type, int batch,
              size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    _q = exec_queue;
    init<int>(dim, n, inembed, istride, idist, input_type, onembed, ostride,
              odist, output_type, batch, direction_and_placement);
  }
  /// Commit the configuration to calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [out] scratchpad_size The workspace size required for this FFT.
  /// If this value is used to allocate memory, \p direction_and_placement need
  /// to be specified explicitly to get correct result.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  void commit(sycl::queue *exec_queue, int dim, int *n, int *inembed,
              int istride, int idist, int *onembed, int ostride, int odist,
              fft_type type, int batch, size_t *scratchpad_size,
              std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                  direction_and_placement = std::nullopt) {
    commit(exec_queue, dim, n, inembed, istride, idist,
           fft_type_to_data_type(type).first, onembed, ostride, odist,
           fft_type_to_data_type(type).second, batch, scratchpad_size,
           direction_and_placement);
  }
  /// Create the class for calculate n-D FFT.
  /// \param [in] exec_queue The queue where the calculation should be executed.
  /// \param [in] dim Dimension number of the data.
  /// \param [in] n Pointer to an array containing each dimension's size.
  /// \param [in] inembed Pointer to an array containing each dimension's size
  /// of the embedded input data.
  /// \param [in] istride Stride size of the input data.
  /// \param [in] idist Distance between the two batches of the input data.
  /// \param [in] onembed Pointer to an array containing each dimension's size
  /// of the embedded output data.
  /// \param [in] ostride Stride size of the output data.
  /// \param [in] odist Distance between the two batches of the output data.
  /// \param [in] type The FFT type.
  /// \param [in] batch The number of FFT operations to perform.
  /// \param [in] direction_and_placement Explicitly specify the FFT direction
  /// and placement info. If this value is specified, the direction parameter
  /// will be ignored in the fft_engine::compute function. If it is not set,
  /// forward direction(if current FFT is complex-to-complex) and out-of-place
  /// (false) are set by default.
  static fft_engine *
  create(sycl::queue *exec_queue, int dim, int *n, int *inembed, int istride,
         int idist, int *onembed, int ostride, int odist, fft_type type,
         int batch,
         std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
             direction_and_placement = std::nullopt) {
    fft_engine *engine = new fft_engine();
    engine->commit(exec_queue, dim, n, inembed, istride, idist, onembed,
                   ostride, odist, type, batch, nullptr,
                   direction_and_placement);
    return engine;
  }

  /// Destroy the class for calculate FFT.
  /// \param [in] engine Pointer returned from fft_engine::craete.
  static void destroy(fft_engine *engine) { delete engine; }


  /// Execute the FFT calculation.
  /// \param [in] input Pointer to the input data.
  /// \param [out] output Pointer to the output data.
  /// \param [in] direction The FFT direction.
  template <typename input_t, typename output_t>
  void compute(input_t *input, output_t *output, fft_direction direction) {
      compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
          (float *)input, (float *)output, direction);
  }
  template <>
  void compute(sycl::float2 *input, sycl::float2 *output,
               fft_direction direction) {
    compute_complex<float, oneapi::mkl::dft::precision::SINGLE>(
        (float *)input, (float *)output, direction);
  }
  /// Setting the user's SYCL queue for calculation.
  /// \param [in] q Pointer to the SYCL queue.
  void set_queue(sycl::queue *q) { _q = q; }

private:
  static std::pair<library_data_t, library_data_t>
  fft_type_to_data_type(fft_type type) {
    switch (type) {
    case fft_type::complex_float_to_complex_float: {
      return std::make_pair(library_data_t::complex_float,
                            library_data_t::complex_float);
    }
    }
  }

  void config_and_commit_basic() {
      _desc_sc = std::make_shared<
          oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                       oneapi::mkl::dft::domain::COMPLEX>>(_n);
      std::int64_t distance = 1;
      for (auto i : _n)
        distance = distance * i;
      _fwd_dist = distance;
      _bwd_dist = distance;
      _desc_sc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                          distance);
      _desc_sc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                          distance);
      _desc_sc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                          _batch);
#ifdef __INTEL_MKL__
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_INPLACE);
      else
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
      if (_use_external_workspace) {
        if (_q->get_device().is_gpu()) {
          _desc_sc->set_value(
              oneapi::mkl::dft::config_param::WORKSPACE,
              oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
        }
      }
      if (_is_estimate_call) {
        if (_q->get_device().is_gpu()) {
          _desc_sc->get_value(
              oneapi::mkl::dft::config_param::WORKSPACE_ESTIMATE_BYTES,
              &_workspace_estimate_bytes);
        }
      } else {
        _desc_sc->commit(*_q);
        if (_q->get_device().is_gpu()) {
          _desc_sc->get_value(oneapi::mkl::dft::config_param::WORKSPACE_BYTES,
                              &_workspace_bytes);
        }
      }
#else
      if (_is_user_specified_dir_and_placement && _is_inplace)
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::INPLACE);
      else
        _desc_sc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                            oneapi::mkl::dft::config_value::NOT_INPLACE);
      _desc_sc->commit(*_q);
#endif
  }

  template <typename T>
  void init(int dim, T *n, T *inembed, T istride, T idist,
            library_data_t input_type, T *onembed, T ostride, T odist,
            library_data_t output_type, T batch,
            std::optional<std::pair<fft_direction, bool /*is_inplace*/>>
                direction_and_placement) {
    if (direction_and_placement.has_value()) {
      _is_user_specified_dir_and_placement = true;
      _direction = direction_and_placement->first;
      _is_inplace = direction_and_placement->second;
    }
    _n.resize(dim);
    _inembed.resize(dim);
    _onembed.resize(dim);
    _input_type = input_type;
    _output_type = output_type;
    for (int i = 0; i < dim; i++) {
      _n[i] = n[i];
    }
    _is_basic = true;  // always true
    _batch = batch;
    _dim = dim;

    config_and_commit_basic();
  }

  template <class Desc_t> void swap_distance(std::shared_ptr<Desc_t> desc) {
    std::swap(_fwd_dist, _bwd_dist);
    desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, _fwd_dist);
    desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, _bwd_dist);
  }

#ifdef __INTEL_MKL__
  template <class Desc_t> void swap_strides(std::shared_ptr<Desc_t> desc) {
    std::swap_ranges(_fwd_strides.begin(), _fwd_strides.end(),
                     _bwd_strides.begin());
    desc->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,
                    _fwd_strides.data());
    desc->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES,
                    _bwd_strides.data());
  }
#endif

  template <bool Is_inplace, class Desc_t>
  void set_stride_and_distance_basic(std::shared_ptr<Desc_t> desc) {
    std::int64_t forward_distance = 0;
    std::int64_t backward_distance = 0;
    if (_dim == 1) {
      if constexpr (Is_inplace) {
        _fwd_strides = {0, 1, 0, 0};
        _bwd_strides = {0, 1, 0, 0};
        forward_distance = 2 * (_n[0] / 2 + 1);
        backward_distance = _n[0] / 2 + 1;
      } else {
        _fwd_strides = {0, 1, 0, 0};
        _bwd_strides = {0, 1, 0, 0};
        forward_distance = _n[0];
        backward_distance = _n[0] / 2 + 1;
      }
    } else if (_dim == 2) {
      if constexpr (Is_inplace) {
        _bwd_strides = {0, _n[1] / 2 + 1, 1, 0};
        _fwd_strides = {0, 2 * (_n[1] / 2 + 1), 1, 0};
        forward_distance = _n[0] * 2 * (_n[1] / 2 + 1);
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      } else {
        _bwd_strides = {0, _n[1] / 2 + 1, 1, 0};
        _fwd_strides = {0, _n[1], 1, 0};
        forward_distance = _n[0] * _n[1];
        backward_distance = _n[0] * (_n[1] / 2 + 1);
      }
    } else if (_dim == 3) {
      if constexpr (Is_inplace) {
        _bwd_strides = {0, _n[1] * (_n[2] / 2 + 1), _n[2] / 2 + 1, 1};
        _fwd_strides = {0, _n[1] * 2 * (_n[2] / 2 + 1), 2 * (_n[2] / 2 + 1), 1};
        forward_distance = _n[0] * _n[1] * 2 * (_n[2] / 2 + 1);
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      } else {
        _bwd_strides = {0, _n[1] * (_n[2] / 2 + 1), _n[2] / 2 + 1, 1};
        _fwd_strides = {0, _n[1] * _n[2], _n[2], 1};
        forward_distance = _n[0] * _n[1] * _n[2];
        backward_distance = _n[0] * _n[1] * (_n[2] / 2 + 1);
      }
    }
#ifdef __INTEL_MKL__
    desc->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,
                    _fwd_strides.data());
    desc->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES,
                    _bwd_strides.data());
#else
    if (_direction == fft_direction::forward) {
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                      _fwd_strides.data());
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                      _bwd_strides.data());
    } else {
      desc->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                      _bwd_strides.data());
      desc->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                      _fwd_strides.data());
    }
#endif
    desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                    forward_distance);
    desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                    backward_distance);
  }

  template <class Dest_t, class T>
  void compute_impl(Dest_t desc, T *input, T *output,
                    std::optional<fft_direction> direction) {
    bool is_this_compute_inplace = input == output;
    constexpr bool is_complex =
        std::is_same_v<Dest_t, std::shared_ptr<oneapi::mkl::dft::descriptor<
                                   oneapi::mkl::dft::precision::SINGLE,
                                   oneapi::mkl::dft::domain::COMPLEX>>> ||
        std::is_same_v<Dest_t, std::shared_ptr<oneapi::mkl::dft::descriptor<
                                   oneapi::mkl::dft::precision::DOUBLE,
                                   oneapi::mkl::dft::domain::COMPLEX>>>;

    if (!_is_user_specified_dir_and_placement) {
      // The descriptor need different config values if the FFT direction
      // or placement is different.
      // Here we check the conditions, and new config values are set and
      // re-committed if needed.
      bool need_commit = false;
      if constexpr (is_complex) {
        if (direction.value() != _direction) {
          need_commit = true;
          swap_distance(desc);
#ifdef __INTEL_MKL__
          if (!_is_basic)
            swap_strides(desc);
#endif
          _direction = direction.value();
        }
      }
      if (is_this_compute_inplace != _is_inplace) {
        need_commit = true;
        _is_inplace = is_this_compute_inplace;
#ifdef __INTEL_MKL__
        if (_is_inplace) {
          desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                          DFTI_CONFIG_VALUE::DFTI_INPLACE);
          if constexpr (!is_complex)
            if (_is_basic)
              set_stride_and_distance_basic<true>(desc);
        } else {
          desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                          DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
          if constexpr (!is_complex)
            if (_is_basic)
              set_stride_and_distance_basic<false>(desc);
        }
#else
        if (_is_inplace) {
          desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                          oneapi::mkl::dft::config_value::INPLACE);
          if constexpr (!is_complex)
            if (_is_basic)
              set_stride_and_distance_basic<true>(desc);
        } else {
          desc->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                          oneapi::mkl::dft::config_value::NOT_INPLACE);
          if constexpr (!is_complex)
            if (_is_basic)
              set_stride_and_distance_basic<false>(desc);
        }
#endif
      }
      if (need_commit)
        desc->commit(*_q);
    }

    if (_is_inplace) {
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward<typename Dest_t::element_type, T>(
            *desc, input);
      } else {
        oneapi::mkl::dft::compute_backward<typename Dest_t::element_type, T>(
            *desc, input);
      }
    } else {
      if (_direction == fft_direction::forward) {
        oneapi::mkl::dft::compute_forward<typename Dest_t::element_type, T, T>(
            *desc, input, output);
      } else {
        oneapi::mkl::dft::compute_backward<typename Dest_t::element_type, T, T>(
            *desc, input, output);
      }
    }
  }

  template <class T, oneapi::mkl::dft::precision Precision>
  void compute_complex(T *input, T *output, fft_direction direction) {
    compute_impl(_desc_sc, input, output, direction);
  }

private:
  sycl::queue *_q = nullptr;
  int _dim;
  std::vector<std::int64_t> _n;
  std::vector<std::int64_t> _inembed;
  std::int64_t _istride;
  std::int64_t _fwd_dist;
  library_data_t _input_type;
  std::vector<std::int64_t> _onembed;
  std::int64_t _ostride;
  std::int64_t _bwd_dist;
  library_data_t _output_type;
  std::int64_t _batch = 1;
  bool _is_basic = false;
  bool _is_inplace = false;
  fft_direction _direction = fft_direction::forward;
  bool _is_user_specified_dir_and_placement = false;
  bool _use_external_workspace = false;
  void *_external_workspace_ptr = nullptr;
  size_t _workspace_bytes = 0;
  bool _is_estimate_call = false;
  size_t _workspace_estimate_bytes = 0;
  std::shared_ptr<oneapi::mkl::dft::descriptor<
      oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>>
      _desc_sc;
  std::array<std::int64_t, 4> _fwd_strides = {0, 0, 0, 0};
  std::array<std::int64_t, 4> _bwd_strides = {0, 0, 0, 0};
};

using fft_engine_ptr = fft_engine *;
} // namespace fft

#endif
