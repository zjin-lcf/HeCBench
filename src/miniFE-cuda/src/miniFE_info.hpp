#ifndef miniFE_info_hpp
#define miniFE_info_hpp

#define MINIFE_HOSTNAME "spark-36ac"
#define MINIFE_KERNEL_NAME "'Linux'"
#define MINIFE_KERNEL_RELEASE "'6.14.0-1013-nvidia'"
#define MINIFE_PROCESSOR "'aarch64'"

#define MINIFE_CXX "'/usr/local/cuda/bin/nvcc'"
#define MINIFE_CXX_VERSION "'nvcc: NVIDIA (R) Cuda compiler driver'"
#define MINIFE_CXXFLAGS "'-I. -I../utils -I../fem -DMINIFE_SCALAR=double -DMINIFE_LOCAL_ORDINAL=int -DMINIFE_GLOBAL_ORDINAL=int -DMINIFE_RESTRICT=__restrict__ -O3 -x cu -arch=sm_121 -DMINIFE_CSR_MATRIX '"

#endif
