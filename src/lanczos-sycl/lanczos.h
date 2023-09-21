#ifndef _LANCZOS_H
#define _LANCZOS_H

#include <vector>

#include "matrix.h"
#include <sycl/sycl.hpp>

using std::vector;

template <typename T>
vector<T> gpu_lanczos_eigen(sycl::queue &q, const csr_matrix<T> &matrix, int k, int steps);

#endif
