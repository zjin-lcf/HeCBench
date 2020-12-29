#ifndef _LANCZOS_H
#define _LANCZOS_H

#include <vector>

#include "matrix.h"
#include "common.h"

using std::vector;

template <typename T>
vector<T> gpu_lanczos_eigen(queue &q, const csr_matrix<T> &matrix,
    int k, int steps);

#endif
