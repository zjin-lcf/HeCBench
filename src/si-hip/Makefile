# si-hip is the only HIP benchmark that builds via CMake instead of Make.
# This wrapper drives the CMake build out-of-tree under release/ and
# matches the all/clean interface the rest of HeCBench expects.
HIP_ARCH ?= gfx906
ROCM_PATH ?= /opt/rocm

# si-hip's CMakeLists.txt looks up ROCM_PATH from the environment to add
# it to CMAKE_PREFIX_PATH, so we export it here.
export ROCM_PATH
export HIP_PATH=$(ROCM_PATH)
export CMAKE_PREFIX_PATH=$(ROCM_PATH)
export HIP_PLATFORM=amd

.PHONY: all clean

all:
	mkdir -p release
	cd release && cmake $(CURDIR) \
	    -DCMAKE_BUILD_TYPE=Release \
	    -DCMAKE_CXX_COMPILER:PATH=$(ROCM_PATH)/bin/hipcc \
	    -DCMAKE_CXX_FLAGS=--offload-arch=$(HIP_ARCH)
	$(MAKE) -C release

clean:
	rm -rf release
