#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
/**
 *
 *  @file cuThomasBatch.h
 *
 *  @brief cuThomasBatch kernel implementaion.
 *
 *  cuThomasBatch is a software package provided by
 *  Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *  @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *  @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

SYCL_EXTERNAL void cuThomasBatch(const double *L, const double *D, double *U,
                                 double *RHS, const int M, const int BATCHCOUNT,
                                 sycl::nd_item<3> item_ct1);
