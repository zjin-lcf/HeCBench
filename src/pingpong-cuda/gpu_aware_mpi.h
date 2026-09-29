#ifndef PINGPONG_GPU_AWARE_MPI_H
#define PINGPONG_GPU_AWARE_MPI_H

// Decide, before any device pointer reaches MPI, whether the MPI library can
// take GPU buffers. Included by the CUDA, HIP, and SYCL main-mpi programs.
//
// Each MPI is asked the way it documents:
//   Cray MPICH    MPICH_GPU_SUPPORT_ENABLED=1 (default off), then the MPICH
//                 query below when the headers have it.
//   Intel MPI     I_MPI_OFFLOAD nonzero (default 0).
//   MVAPICH2      MV2_USE_CUDA / MV2_USE_ROCM (unset: cannot tell).
//   MPICH >= 4.0.1 (and derivatives)
//                 MPIX_Query_cuda_support / _hip_support / _ze_support. They
//                 honor MPIR_CVAR_ENABLE_GPU.
//   Open MPI      MPIX_Query_cuda_support (CUDA extension in mpi-ext.h) and
//                 MPIX_Query_rocm_support (ROCm extension, Open MPI 5).
//
// A "no" always aborts. When the library cannot tell (for example Open MPI
// 4.1 + UCX on ROCm, or an MPI not listed above), the program also aborts
// unless MPI_GPU_AWARE=1 is set to assert that the MPI is GPU-aware.
//
// MPIX_GPU_SUPPORT_CUDA / _ZE / _HIP in MPICH are type ids (0, 1, 2), not
// feature flags, so they cannot guard the calls. Gate on the MPICH version
// that added all three query functions instead.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// MPICH 4.0.1 in MPICH_NUMVERSION (major*1e7 + minor*1e5 + rev*1e3 +
// release_type*100 + patch, with a regular release type of 3).
#define PINGPONG_MPICH_GPU_QUERY_VERSION 40001300

#if defined(MPICH_NUMVERSION) && (MPICH_NUMVERSION >= PINGPONG_MPICH_GPU_QUERY_VERSION)
#  define PINGPONG_HAVE_MPICH_GPU_QUERY 1
#endif

typedef struct {
  int answer;
  const char *why;   // what decided the answer
  const char *hint;  // how to turn GPU support on, if known
} pingpong_gpu_aware_t;

static const char *pingpong_gpu_kind_name(int kind)
{
  if (kind == PINGPONG_GPU_KIND_CUDA)
    return "CUDA";
  if (kind == PINGPONG_GPU_KIND_HIP)
    return "HIP";
  if (kind == PINGPONG_GPU_KIND_ZE)
    return "Level Zero";
  return "unknown-vendor";
}

static int pingpong_library_version_has(const char *needle)
{
  char version[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  if (MPI_Get_library_version(version, &length) != MPI_SUCCESS)
    return 0;
  if (length < 0)
    length = 0;
  if (length >= MPI_MAX_LIBRARY_VERSION_STRING)
    length = MPI_MAX_LIBRARY_VERSION_STRING - 1;
  version[length] = '\0';
  return strstr(version, needle) != NULL;
}

// 1 when the variable is set to a nonzero integer, 0 when set to zero,
// -1 when unset or empty.
static int pingpong_env_flag(const char *name)
{
  const char *value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return -1;
  return atoi(value) != 0;
}

static pingpong_gpu_aware_t pingpong_result(int answer, const char *why,
                                            const char *hint)
{
  pingpong_gpu_aware_t r;
  r.answer = answer;
  r.why = why;
  r.hint = hint;
  return r;
}

#if defined(PINGPONG_HAVE_MPICH_GPU_QUERY)
static pingpong_gpu_aware_t pingpong_mpich_query(int kind, const char *hint)
{
  if (kind == PINGPONG_GPU_KIND_CUDA)
    return MPIX_Query_cuda_support() == 1
               ? pingpong_result(PINGPONG_GPU_AWARE_YES, "MPIX_Query_cuda_support() returned 1", hint)
               : pingpong_result(PINGPONG_GPU_AWARE_NO, "MPIX_Query_cuda_support() returned 0", hint);
  if (kind == PINGPONG_GPU_KIND_HIP)
    return MPIX_Query_hip_support() == 1
               ? pingpong_result(PINGPONG_GPU_AWARE_YES, "MPIX_Query_hip_support() returned 1", hint)
               : pingpong_result(PINGPONG_GPU_AWARE_NO, "MPIX_Query_hip_support() returned 0", hint);
  if (kind == PINGPONG_GPU_KIND_ZE)
    return MPIX_Query_ze_support() == 1
               ? pingpong_result(PINGPONG_GPU_AWARE_YES, "MPIX_Query_ze_support() returned 1", hint)
               : pingpong_result(PINGPONG_GPU_AWARE_NO, "MPIX_Query_ze_support() returned 0", hint);
  return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN, "MPICH has no query for this GPU vendor", NULL);
}
#endif

static pingpong_gpu_aware_t pingpong_cray_mpich(int kind)
{
  const char *hint = "set MPICH_GPU_SUPPORT_ENABLED=1 and link the Cray GTL library "
                     "(libmpi_gtl_cuda or libmpi_gtl_hsa)";
  if (pingpong_env_flag("MPICH_GPU_SUPPORT_ENABLED") != 1)
    return pingpong_result(PINGPONG_GPU_AWARE_NO,
                           "Cray MPICH with MPICH_GPU_SUPPORT_ENABLED unset or 0", hint);
#if defined(PINGPONG_HAVE_MPICH_GPU_QUERY)
  return pingpong_mpich_query(kind, hint);
#else
  (void)kind;
  return pingpong_result(PINGPONG_GPU_AWARE_YES, "Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1", hint);
#endif
}

static pingpong_gpu_aware_t pingpong_intel_mpi(void)
{
  const char *hint = "set I_MPI_OFFLOAD=1";
  if (pingpong_env_flag("I_MPI_OFFLOAD") == 1)
    return pingpong_result(PINGPONG_GPU_AWARE_YES, "Intel MPI with I_MPI_OFFLOAD nonzero", hint);
  return pingpong_result(PINGPONG_GPU_AWARE_NO, "Intel MPI with I_MPI_OFFLOAD unset or 0", hint);
}

static pingpong_gpu_aware_t pingpong_mvapich2(int kind)
{
  const char *name = NULL;
  const char *hint = NULL;
  if (kind == PINGPONG_GPU_KIND_CUDA) {
    name = "MV2_USE_CUDA";
    hint = "set MV2_USE_CUDA=1 (MVAPICH2-GDR or a CUDA-enabled MVAPICH2 build)";
  } else if (kind == PINGPONG_GPU_KIND_HIP) {
    name = "MV2_USE_ROCM";
    hint = "set MV2_USE_ROCM=1 (MVAPICH2-GDR built with ROCm)";
  } else {
    return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                           "MVAPICH2 has no switch for this GPU vendor", NULL);
  }
  const int flag = pingpong_env_flag(name);
  if (flag == 1)
    return pingpong_result(PINGPONG_GPU_AWARE_YES,
                           kind == PINGPONG_GPU_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA=1"
                                                          : "MVAPICH2 with MV2_USE_ROCM=1",
                           hint);
  if (flag == 0)
    return pingpong_result(PINGPONG_GPU_AWARE_NO,
                           kind == PINGPONG_GPU_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA=0"
                                                          : "MVAPICH2 with MV2_USE_ROCM=0",
                           hint);
  return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                         kind == PINGPONG_GPU_KIND_CUDA ? "MVAPICH2 with MV2_USE_CUDA unset"
                                                        : "MVAPICH2 with MV2_USE_ROCM unset",
                         hint);
}

#if defined(OPEN_MPI) && OPEN_MPI
static pingpong_gpu_aware_t pingpong_openmpi(int kind)
{
  if (kind == PINGPONG_GPU_KIND_CUDA) {
    const char *hint = "use an Open MPI built with CUDA support (for example the NVIDIA HPC-X or HPC SDK MPI)";
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA || defined(MPIX_CUDA_AWARE_SUPPORT)
    return MPIX_Query_cuda_support() == 1
               ? pingpong_result(PINGPONG_GPU_AWARE_YES, "MPIX_Query_cuda_support() returned 1", hint)
               : pingpong_result(PINGPONG_GPU_AWARE_NO, "MPIX_Query_cuda_support() returned 0", hint);
#else
    return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                           "Open MPI headers have no CUDA extension (mpi-ext.h)", hint);
#endif
  }
  if (kind == PINGPONG_GPU_KIND_HIP) {
    const char *hint = "use an Open MPI built with ROCm support (Open MPI 5 reports it; "
                       "Open MPI 4 with ROCm-enabled UCX cannot report it)";
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM || defined(MPIX_ROCM_AWARE_SUPPORT)
    return MPIX_Query_rocm_support() == 1
               ? pingpong_result(PINGPONG_GPU_AWARE_YES, "MPIX_Query_rocm_support() returned 1", hint)
               : pingpong_result(PINGPONG_GPU_AWARE_NO, "MPIX_Query_rocm_support() returned 0", hint);
#else
    return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                           "Open MPI headers have no ROCm extension (mpi-ext.h, Open MPI 5+)", hint);
#endif
  }
  return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                         "Open MPI has no query for this GPU vendor", NULL);
}
#endif

// Call after MPI_Init. Does not abort.
static pingpong_gpu_aware_t pingpong_query_gpu_aware(int kind)
{
  if (kind == PINGPONG_GPU_KIND_UNKNOWN)
    return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN, "the GPU vendor is not recognized", NULL);

  // Runtime library checks first: these MPIs are MPICH-derived, and their
  // own switch decides even when the MPICH query is in the headers.
  if (pingpong_library_version_has("CRAY MPICH"))
    return pingpong_cray_mpich(kind);
  if (pingpong_library_version_has("Intel(R) MPI") ||
      pingpong_library_version_has("Intel MPI"))
    return pingpong_intel_mpi();
  if (pingpong_library_version_has("MVAPICH2"))
    return pingpong_mvapich2(kind);

#if defined(OPEN_MPI) && OPEN_MPI
  return pingpong_openmpi(kind);
#elif defined(PINGPONG_HAVE_MPICH_GPU_QUERY)
  return pingpong_mpich_query(kind, "enable GPU support in MPICH (MPIR_CVAR_ENABLE_GPU=1, "
                                    "and a build configured with CUDA, HIP, or Level Zero)");
#else
  return pingpong_result(PINGPONG_GPU_AWARE_UNKNOWN,
                         "this MPI has no GPU-buffer query (not Cray MPICH, Intel MPI, "
                         "MVAPICH2, Open MPI, or MPICH >= 4.0.1)", NULL);
#endif
}

// Abort unless the MPI library reports GPU-buffer support, or it cannot tell
// and MPI_GPU_AWARE=1 is set.
static void pingpong_require_gpu_aware_mpi(int kind, int rank)
{
  const pingpong_gpu_aware_t r = pingpong_query_gpu_aware(kind);
  const char *kind_name = pingpong_gpu_kind_name(kind);

  if (r.answer == PINGPONG_GPU_AWARE_YES)
    return;

  if (r.answer == PINGPONG_GPU_AWARE_UNKNOWN && pingpong_env_flag("MPI_GPU_AWARE") == 1) {
    if (rank == 0) {
      printf("WARNING: MPI cannot report %s GPU-buffer support (%s); "
             "continuing because MPI_GPU_AWARE=1.\n", kind_name, r.why);
      fflush(stdout);
    }
    return;
  }

  if (r.answer == PINGPONG_GPU_AWARE_NO) {
    fprintf(stderr,
            "ERROR: rank %d: MPI library reports no %s GPU-buffer support (%s).\n",
            rank, kind_name, r.why);
  } else {
    fprintf(stderr,
            "ERROR: rank %d: MPI library cannot report %s GPU-buffer support (%s).\n"
            "If this MPI is GPU-aware, set MPI_GPU_AWARE=1 to continue.\n",
            rank, kind_name, r.why);
  }
  if (r.hint)
    fprintf(stderr, "To enable GPU buffers: %s.\n", r.hint);
  fprintf(stderr, "main-mpi passes device pointers to MPI_Send and MPI_Recv.\n");
  fflush(stderr);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#endif
