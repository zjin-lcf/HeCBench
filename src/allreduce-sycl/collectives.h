#ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_
#define BAIDU_ALLREDUCE_COLLECTIVES_H_ value

#include <cstddef>
#include <vector>
#include <mpi.h>
#include <sycl/sycl.hpp>

/*
 * This file contains the implementation of the baidu-allreduce communication
 * collectives, and provides the following functions:
 *
 *    void InitCollectives(int device);
 *    void RingAllreduce(float* data, size_t length, float** output);
 *    void RingAllgather(float* data, size_t length, float** output);
 *
 */

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
void InitCollectives(int device, const std::vector<sycl::device> &devices);

// The ring allreduce. The lengths of the data chunks passed to this function
// must be the same across all MPI processes. The output memory will be
// allocated and written into `output`.
void RingAllreduce(float* data, size_t length, float** output);

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
void RingAllgather(float* data, size_t length, float** output);

#endif /* ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_ */
