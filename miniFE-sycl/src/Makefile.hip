#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int \
	-DMINIFE_RESTRICT=__restrict__

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3
CXXFLAGS = $(CFLAGS)

MARCH     = gfx906
PLATFORM  = rocm
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) -I../../include/ \
	   -DUSE_GPU \
           --hipsycl-platform=$(PLATFORM) \
	   --hipsycl-gpu-arch=$(MARCH) \
	   $(ROCM_DEVICE_LIB) \
	   $(MINIFE_MATRIX_TYPE) \
#            -DMINIFE_DEBUG \
#	-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK \
#	-DMINIFE_REPORT_RUSAGE

LDFLAGS=$(CFLAGS)
LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

CXX=syclcc
CC=syclcc


include make_targets
