#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = syclcc
OPTIMIZE  = yes
DEBUG     = no
MARCH     = gfx906
PLATFORM  = rocm
DEVICE    = gpu
ROCM_DEVICE_LIB = -nogpulib \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ocml.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_daz_opt_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_finite_only_on.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/oclc_isa_version_906.bc \
	     -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/hip.bc


#===============================================================================
# Program name & source code list
#===============================================================================

program = s3d

source = main.cpp  Option.cpp  OptionParser.cpp  S3D.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH) \
	  $(ROCM_DEVICE_LIB) 

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DUSE_GPU
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp OptionParser.h Utility.h 
	$(CC) $(CFLAGS) -c $< -o $@

Option.o: Option.cpp Option.h 
	$(CC) $(CFLAGS) -c $< -o $@

OptionParser.o: OptionParser.cpp OptionParser.h  Utility.h
	$(CC) $(CFLAGS) -c $< -o $@

S3D.o: S3D.cpp S3D.h OptionParser.h *.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) -q -n 100 -s 1

