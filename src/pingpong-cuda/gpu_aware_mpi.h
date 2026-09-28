#ifndef PINGPONG_GPU_AWARE_MPI_H
#define PINGPONG_GPU_AWARE_MPI_H

// Ask the MPI library whether it can take a device pointer, then abort when
// the library itself says no. This header is included by the CUDA, HIP, and
// SYCL main-mpi programs.
//
// The calls are the ones each library documents:
//   CUDA:  MPIX_Query_cuda_support()
//          Open MPI declares it from mpi-ext.h (MPIX_CUDA_AWARE_SUPPORT /
//          OMPI_HAVE_MPI_EXT_CUDA). MPICH >= 4.0.1 declares it from mpi.h.
//   HIP:   MPIX_Query_hip_support() on MPICH >= 4.0.1, or
//          MPIX_Query_rocm_support() on Open MPI (MPIX_ROCM_AWARE_SUPPORT /
//          OMPI_HAVE_MPI_EXT_ROCM, from v5). Open MPI 4 has neither, so the
//          caller still does one device-pointer check.
//   ZE:    MPIX_Query_ze_support() on MPICH >= 4.0.1. Intel MPI uses its own
//          I_MPI_OFFLOAD switch (default 0) instead of that query.
//
// MPIX_GPU_SUPPORT_CUDA / _ZE / _HIP in MPICH are type ids (0, 1, 2), not
// feature flags. CUDA's id is 0, so "#if MPIX_GPU_SUPPORT_CUDA" is false on
// every MPICH that has the API. Gate the MPICH calls on the library version
// that added all three functions (4.0.1).

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(OPEN_MPI) && OPEN_MPI
#  if defined(__has_include)
#    if __has_include(<mpi-ext.h>)
#      include <mpi-ext.h>
#    endif
#  else
#    include <mpi-ext.h>
#  endif
#endif

#define PINGPONG_GPU_KIND_UNKNOWN 0
#define PINGPONG_GPU_KIND_CUDA 1
#define PINGPONG_GPU_KIND_HIP 2
#define PINGPONG_GPU_KIND_ZE 3

#define PINGPONG_GPU_AWARE_NO 0
#define PINGPONG_GPU_AWARE_YES 1
#define PINGPONG_GPU_AWARE_UNKNOWN (-1)

// MPICH 4.0.1 is 40001300 in MPICH_NUMVERSION (major*1e7 + minor*1e5 +
// rev*1e3 + release_type*100 + patch, with a regular release type of 3).
#define PINGPONG_MPICH_GPU_QUERY_VERSION 40001300

static int pingpong_library_is_intel_mpi(void)
{
#if !defined(MPI_MAX_LIBRARY_VERSION_STRING)
  return 0;
#else
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  if (MPI_Get_library_version(version, &length) != MPI_SUCCESS)
    return 0;
  if (length < 0)
    length = 0;
  if (length >= MPI_MAX_LIBRARY_VERSION_STRING)
    length = MPI_MAX_LIBRARY_VERSION_STRING - 1;
  version[length] = '\0';
  if (strstr(version, "Intel(R) MPI") != NULL)
    return 1;
  if (strstr(version, "Intel MPI") != NULL)
    return 1;
  return 0;
#endif
}

// Intel MPI enables GPU buffers only when I_MPI_OFFLOAD is nonzero. The
// default is 0. Returns 1 when this library is Intel MPI and *answer is set.
static int pingpong_intel_mpi_gpu(int *answer, const char **why)
{
  if (!pingpong_library_is_intel_mpi())
    return 0;
  const char *off = getenv("I_MPI_OFFLOAD");
  if (off != NULL && atoi(off) != 0) {
    *answer = PINGPONG_GPU_AWARE_YES;
    return 1;
  }
  if (why)
    *why = " (Intel MPI I_MPI_OFFLOAD is unset or 0)";
  *answer = PINGPONG_GPU_AWARE_NO;
  return 1;
}

#if defined(OPEN_MPI) && OPEN_MPI
static int pingpong_openmpi_cuda(const char **why)
{
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
  if (MPIX_Query_cuda_support() == 1)
    return PINGPONG_GPU_AWARE_YES;
  if (why)
    *why = " (MPIX_Query_cuda_support returned 0)";
  return PINGPONG_GPU_AWARE_NO;
#elif defined(OMPI_HAVE_MPI_EXT_CUDA)
  if (why)
    *why = " (Open MPI CUDA extension is disabled)";
  return PINGPONG_GPU_AWARE_NO;
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
  // Declared beside the macro, including when the macro is 0. The function
  // returns 0 unless this build can take CUDA pointers.
  if (MPIX_Query_cuda_support() == 1)
    return PINGPONG_GPU_AWARE_YES;
  if (why)
    *why = " (MPIX_Query_cuda_support returned 0)";
  return PINGPONG_GPU_AWARE_NO;
#else
  (void)why;
  return PINGPONG_GPU_AWARE_UNKNOWN;
#endif
}
#endif

#if defined(OPEN_MPI) && OPEN_MPI
static int pingpong_openmpi_rocm(const char **why)
{
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
  if (MPIX_Query_rocm_support() == 1)
    return PINGPONG_GPU_AWARE_YES;
  if (why)
    *why = " (MPIX_Query_rocm_support returned 0)";
  return PINGPONG_GPU_AWARE_NO;
#elif defined(OMPI_HAVE_MPI_EXT_ROCM)
  if (why)
    *why = " (Open MPI ROCm extension is disabled)";
  return PINGPONG_GPU_AWARE_NO;
#elif defined(MPIX_ROCM_AWARE_SUPPORT)
  if (MPIX_Query_rocm_support() == 1)
    return PINGPONG_GPU_AWARE_YES;
  if (why)
    *why = " (MPIX_Query_rocm_support returned 0)";
  return PINGPONG_GPU_AWARE_NO;
#else
  // Open MPI 4 has no ROCm query. The device-pointer check still runs.
  (void)why;
  return PINGPONG_GPU_AWARE_UNKNOWN;
#endif
}
#endif

#if defined(MPICH_NUMVERSION) && (MPICH_NUMVERSION >= PINGPONG_MPICH_GPU_QUERY_VERSION)
static int pingpong_mpich_query(int kind, const char **why)
{
  int supported = 0;
  const char *returned0 = NULL;
  if (kind == PINGPONG_GPU_KIND_CUDA) {
    supported = MPIX_Query_cuda_support();
    returned0 = " (MPIX_Query_cuda_support returned 0)";
  } else if (kind == PINGPONG_GPU_KIND_HIP) {
    supported = MPIX_Query_hip_support();
    returned0 = " (MPIX_Query_hip_support returned 0)";
  } else if (kind == PINGPONG_GPU_KIND_ZE) {
    supported = MPIX_Query_ze_support();
    returned0 = " (MPIX_Query_ze_support returned 0)";
  } else {
    return PINGPONG_GPU_AWARE_UNKNOWN;
  }
  if (supported == 1)
    return PINGPONG_GPU_AWARE_YES;
  if (why)
    *why = returned0;
  return PINGPONG_GPU_AWARE_NO;
}
#endif

static const char *pingpong_gpu_kind_name(int kind)
{
  if (kind == PINGPONG_GPU_KIND_CUDA)
    return "CUDA";
  if (kind == PINGPONG_GPU_KIND_HIP)
    return "HIP";
  if (kind == PINGPONG_GPU_KIND_ZE)
    return "Level Zero";
  return "GPU";
}

// YES, NO, or UNKNOWN. Does not abort. Call after MPI_Init.
static int pingpong_query_gpu_aware(int kind, const char **why)
{
  if (why)
    *why = "";
  int intel_answer = PINGPONG_GPU_AWARE_UNKNOWN;
  if (pingpong_intel_mpi_gpu(&intel_answer, why))
    return intel_answer;

#if defined(OPEN_MPI) && OPEN_MPI
  if (kind == PINGPONG_GPU_KIND_CUDA)
    return pingpong_openmpi_cuda(why);
  if (kind == PINGPONG_GPU_KIND_HIP)
    return pingpong_openmpi_rocm(why);
  (void)kind;
  return PINGPONG_GPU_AWARE_UNKNOWN;
#elif defined(MPICH_NUMVERSION) && (MPICH_NUMVERSION >= PINGPONG_MPICH_GPU_QUERY_VERSION)
  return pingpong_mpich_query(kind, why);
#else
  (void)kind;
  return PINGPONG_GPU_AWARE_UNKNOWN;
#endif
}

// Abort before any device pointer is passed to MPI when the library reports
// no support. Return when it reports support or has no query for this kind.
static void pingpong_require_gpu_aware_mpi(int kind, int rank)
{
  const char *why = "";
  const int answer = pingpong_query_gpu_aware(kind, &why);
  if (answer == PINGPONG_GPU_AWARE_NO) {
    fprintf(stderr,
            "ERROR: MPI library reports no %s GPU-buffer support%s.\n"
            "main-mpi passes device pointers to MPI_Send and MPI_Recv.\n",
            pingpong_gpu_kind_name(kind), why);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
    _exit(1);
  }
  if (answer == PINGPONG_GPU_AWARE_UNKNOWN && rank == 0) {
    printf("MPI library has no %s GPU-buffer query; checking one device-pointer transfer.\n",
           pingpong_gpu_kind_name(kind));
    fflush(stdout);
  }
}

#endif
