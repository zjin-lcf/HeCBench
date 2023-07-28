// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace facebook { namespace cuda {

/// Our tensor type
template <typename T,
          int Dim,
          typename IndexT,
          template <typename U> class PtrTraits>
class DeviceTensor;

/// Type of a subspace of a tensor
namespace detail {
template <typename TensorType,
          int SubDim,
          template <typename U> class PtrTraits>
class DeviceSubTensor;
}

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict PtrType;
};

template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

- `T` is the contained type (e.g., `float`)
- `Dim` is the tensor rank
- `IndexT` is the integer type used for size/stride arrays, and for
- all indexing math. Default is `int`, but for large tensors, `long`
- can be used instead.
- `PtrTraits` are traits applied to our data pointer (T*). By default,
- this is just T*, but RestrictPtrTraits can be used to apply T*
- __restrict for alias-free analysis.
*/
template <typename T,
          int Dim,
          typename IndexT = int,
          template <typename U> class PtrTraits = DefaultPtrTraits>
class DeviceTensor {
 public:
  enum { NumDim = Dim };
  typedef T DataType;
  typedef IndexT IndexType;
  typedef typename PtrTraits<T>::PtrType DataPtrType;
  typedef DeviceTensor<T, Dim, IndexT, PtrTraits> TensorType;

  /// Default constructor
  DeviceTensor();

  /// Constructor that calculates strides with no padding
  DeviceTensor(DataPtrType data,
                                   const IndexT sizes[Dim]);

  /// Constructor that takes arbitrary size/stride arrays
  DeviceTensor(DataPtrType data,
                                   const IndexT sizes[Dim],
                                   const IndexT strides[Dim]);

  /// Returns true if the two tensors are of the same dimensionality
  /// and size.
  template <int OtherDim>
  bool
  isSameSize(
    const DeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const;

  /// Returns true if the two tensors are of the same dimensionality,
  /// size and stride.
  template <int OtherDim>
  bool
  isSameSizeAndStride(
    const DeviceTensor<T, OtherDim, IndexT, PtrTraits>& rhs) const;

  /// Produces a string containing our size and stride array contents;
  /// for debugging purposes
  std::string toString() const;

  /// Cast to a tensor of a different type of the same size and stride
  template <typename U>
  DeviceTensor<U, Dim, IndexT, PtrTraits> cast();

  template <typename U>
  
  const DeviceTensor<U, Dim, IndexT, PtrTraits> cast() const;

  /// Returns a raw pointer to the start of our data.
  inline DataPtrType data() {
    return data_;
  }

  /// Returns a raw pointer to the start of our data (const).
  inline const DataPtrType data() const {
    return data_;
  }

  /// Cast to a different datatype
  template <typename U>
  inline typename PtrTraits<U>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<U>::PtrType>(data_);
  }

  /// Cast to a different datatype
  template <typename U>
  inline const typename PtrTraits<const U>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const U>::PtrType>(data_);
  }

  /// Returns a read/write view of a portion of our tensor.
  inline detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>
  operator[](IndexT);

  /// Returns a read/write view of a portion of our tensor (const).
  inline const detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>
  operator[](IndexT) const;

  /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds
  /// checking.
  inline int getSize(int i) const {
    return size_[i];
  }

  /// Returns the stride of a given dimension, `[0, Dim - 1]`. No bounds
  /// checking.
  inline int getStride(int i) const {
    return stride_[i];
  }

  /// Returns the total number of elements contained within our data
  /// (product of `getSize(i)`)
  long numElements() const;

  /// Returns the size array.
  inline const IndexT *sizes() const {
    return size_;
  }

  /// Returns the stride array.
  inline const IndexT *strides() const {
    return stride_;
  }

  /// Limited form of resize by permutation, make sure your permutation array
  /// is legit. Only works for contiguous tensors.
  void permuteDims(const std::vector<int>& perm);

  /// Returns true if there is no padding within the tensor and no
  /// re-ordering of the dimensions.
  /// ~~~
  /// (stride(i) == size(i + 1) * stride(i + 1))
  /// ~~~
  bool isContiguous() const;

  /// Returns whether a given dimension has only increasing stride
  /// from the previous dimension. A tensor that was permuted by
  /// exchanging size and stride only will fail this check.
  /// If `i == 0` just check `size > 0`. Returns `false` if `stride` is `<= 0`.
  bool isConsistentlySized(int i) const;

  // Returns whether at each dimension `stride <= size`.
  // If this is not the case then iterating once over the size space will
  // touch the same memory locations multiple times.
  bool isConsistentlySized() const;

  /// Returns true if the given dimension index has no padding
  bool isContiguousDim(int i) const;

  /// Returns a tensor of the same dimension after transposing the two
  /// dimensions given. Does not actually move elements; transposition
  /// is made by permuting the size/stride arrays.
  DeviceTensor<T, Dim, IndexT, PtrTraits>
  transpose(int dim1, int dim2) const;

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the leading dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[1][1][2][3]`
  template <int NewDim>
  DeviceTensor<T, NewDim, IndexT, PtrTraits> upcastOuter();

  /// Upcast a tensor of dimension `D` to some tensor of dimension
  /// D' > D by padding the lowest/most varying dimensions by 1
  /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[2][3][1][1]`
  template <int NewDim>
  DeviceTensor<T, NewDim, IndexT, PtrTraits> upcastInner();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  
  DeviceTensor<T, NewDim, IndexT, PtrTraits> downcastOuter();

  /// Downcast a tensor of dimension `D` to some tensor of dimension
  /// D' < D by collapsing the leading dimensions. asserts if there is
  /// padding on the leading dimensions.
  template <int NewDim>
  
  DeviceTensor<T, NewDim, IndexT, PtrTraits> downcastInner();

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting at `at`.
  template <int SubDim>
  DeviceTensor<T, SubDim, IndexT, PtrTraits>
  view(DataPtrType at);

  /// Returns a tensor that is a view of the `SubDim`-dimensional slice
  /// of this tensor, starting where our data begins
  template <int SubDim>
  DeviceTensor<T, SubDim, IndexT, PtrTraits>
  view();

  /// Zeroes out the tensor asynchronously. Asserts if the contents
  /// in question are not contiguous.
  void zero(sycl::queue *stream = 0);

 private:
  /// Raw pointer to where the tensor data begins
  DataPtrType data_;

  /// Array of strides (in sizeof(T) terms) per each dimension
  IndexT stride_[Dim];

  /// Size per each dimension
  IndexT size_[Dim];
};

namespace detail {

/// Specialization for a view of a single value (0-dimensional)
template <typename TensorType, template <typename U> class PtrTraits>
class DeviceSubTensor<TensorType, 0, PtrTraits> {
 public:
  DeviceSubTensor<TensorType, 0, PtrTraits>
  operator=(typename TensorType::DataType val) {
    *data_ = val;
    return *this;
  }

  // operator T&
  operator typename TensorType::DataType&() {
    return *data_;
  }

  // const operator T& returning const T&
  operator const typename TensorType::DataType&() const {
    return *data_;
  }

  // operator& returning T*
  typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  inline typename TensorType::DataPtrType data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  inline const typename TensorType::DataPtrType data() const {
    return data_;
  }

  /// Cast to a different datatype.
  template <typename T>
  T& as() {
    return *dataAs<T>();
  }

  /// Cast to a different datatype (const).
  template <typename T>
  const T& as() const {
    return *dataAs<T>();
  }

  /// Cast to a different datatype
  template <typename T>
  inline typename PtrTraits<T>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  inline typename PtrTraits<const T>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  inline typename TensorType::DataType ldg() const {
    return *data_;
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T> inline T ldgAs() const {
    return as<T>();
  }

  private:
  /// One dimension greater can create us
  friend class DeviceSubTensor<TensorType, 1, PtrTraits>;

  /// Our parent tensor can create us
  friend class DeviceTensor<typename TensorType::DataType,
                            1,
                            typename TensorType::IndexType,
                            PtrTraits>;

  inline DeviceSubTensor(TensorType &t,
                                  typename TensorType::DataPtrType data)
      : tensor_(t), data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// Where our value is located
  typename TensorType::DataPtrType const data_;
};

/// A `SubDim`-rank slice of a parent DeviceTensor
template <typename TensorType,
          int SubDim,
          template <typename U> class PtrTraits>
class DeviceSubTensor {
 public:
  /// Returns a view of the data located at our offset (the dimension
  /// `SubDim` - 1 tensor).
  inline DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
  operator[](typename TensorType::IndexType index) {
    return DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  /// Returns a view of the data located at our offset (the dimension
  /// `SubDim` - 1 tensor) (const).
  inline const DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>
  operator[](typename TensorType::IndexType index) const {
    return DeviceSubTensor<TensorType, SubDim - 1, PtrTraits>(
      tensor_,
      data_ + index * tensor_.getStride(TensorType::NumDim - SubDim));
  }

  // operator& returning T*
  typename TensorType::DataType* operator&() {
    return data_;
  }

  // const operator& returning const T*
  const typename TensorType::DataType* operator&() const {
    return data_;
  }

  /// Returns a raw accessor to our slice.
  inline typename TensorType::DataPtrType data() {
    return data_;
  }

  /// Returns a raw accessor to our slice (const).
  inline const typename TensorType::DataPtrType data() const {
    return data_;
  }

    /// Cast to a different datatype.
  template <typename T>
  T& as() {
    return *dataAs<T>();
  }

  /// Cast to a different datatype (const).
  template <typename T>
  const T& as() const {
    return *dataAs<T>();
  }

  /// Cast to a different datatype
  template <typename T>
  inline typename PtrTraits<T>::PtrType dataAs() {
    return reinterpret_cast<typename PtrTraits<T>::PtrType>(data_);
  }

  /// Cast to a different datatype (const)
  template <typename T>
  inline typename PtrTraits<const T>::PtrType dataAs() const {
    return reinterpret_cast<typename PtrTraits<const T>::PtrType>(data_);
  }

  /// Use the texture cache for reads
  inline typename TensorType::DataType ldg() const {
    return *data_;
  }

  /// Use the texture cache for reads; cast as a particular type
  template <typename T> inline T ldgAs() const {
    return as<T>();
  }

  /// Returns a tensor that is a view of the SubDim-dimensional slice
  /// of this tensor, starting where our data begins
  DeviceTensor<typename TensorType::DataType,
               SubDim,
               typename TensorType::IndexType,
               PtrTraits> view() {
    return tensor_.template view<SubDim>(data_);
  }

 private:
  /// One dimension greater can create us
  friend class DeviceSubTensor<TensorType, SubDim + 1, PtrTraits>;

  /// Our parent tensor can create us
  friend class
  DeviceTensor<typename TensorType::DataType,
               TensorType::NumDim,
               typename TensorType::IndexType,
               PtrTraits>;

  inline DeviceSubTensor(TensorType &t,
                                  typename TensorType::DataPtrType data)
      : tensor_(t), data_(data) {
  }

  /// The tensor we're referencing
  TensorType& tensor_;

  /// The start of our sub-region
  typename TensorType::DataPtrType const data_;
};

} // namespace detail

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits>
inline detail::DeviceSubTensor<DeviceTensor<T, Dim, IndexT, PtrTraits>,
                                        Dim - 1, PtrTraits>
DeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) {
  return detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
    detail::DeviceSubTensor<TensorType, Dim, PtrTraits>(
      *this, data_)[index]);
}

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits>
inline const
    detail::DeviceSubTensor<DeviceTensor<T, Dim, IndexT, PtrTraits>, Dim - 1,
                            PtrTraits>
    DeviceTensor<T, Dim, IndexT, PtrTraits>::operator[](IndexT index) const {
  return detail::DeviceSubTensor<TensorType, Dim - 1, PtrTraits>(
    detail::DeviceSubTensor<TensorType, Dim, PtrTraits>(
      const_cast<TensorType&>(*this), data_)[index]);
}

/// Streaming operator for logging
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
std::ostream& operator<<(
  std::ostream& os, const DeviceTensor<T, Dim, IndexT, PtrTraits>& t) {
  os << t.toString();
  return os;
}

} } // namespace

#include "cuda/DeviceTensor-inl.dp.hpp"
