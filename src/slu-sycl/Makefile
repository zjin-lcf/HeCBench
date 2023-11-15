#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = yes
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================
SLU_PATH = ../slu-cuda/src

SLU_LIBS = $(SLU_PATH)/nicslu/lib/nicslu.a \
           $(SLU_PATH)/nicslu/util/nicslu_util.a

program = main

obj = preprocess.o Timer.o numeric.o symbolic.o main.o $(SLU_LIBS)

#===============================================================================
# Sets Flags
#===============================================================================
SLU_INC = -I$(SLU_PATH)/../include \
          -I$(SLU_PATH)/nicslu/include \
          -I$(SLU_PATH)/nicslu/util

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall $(SLU_INC) -fsycl

# Linker Flags
LDFLAGS = -lpthread

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif

ifeq ($(VERIFY),yes)
  CFLAGS +=-DVERIFY
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

Timer.o: $(SLU_PATH)/Timer.cpp
	$(CC) $(CFLAGS) -c $< -o $@

numeric.o: numeric.cpp
	$(CC) $(CFLAGS) -c $< -o $@

symbolic.o: $(SLU_PATH)/symbolic.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o: $(SLU_PATH)/main.cpp
	$(CC) $(CFLAGS) -c $< -o $@

preprocess.o: $(SLU_PATH)/preprocess.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(SLU_PATH)/nicslu/lib/nicslu.a:
	make -C $(SLU_PATH)/nicslu/

$(SLU_PATH)/nicslu/util/nicslu_util.a:
	make -C $(SLU_PATH)/nicslu/

clean:
	rm -rf $(program) $(obj) *.dat
	make -C $(SLU_PATH)/nicslu/ clean

test: $(program)
	$(LAUNCHER) ./$(program) -i $(SLU_PATH)/nicslu/test/add32.mtx

run: $(program)
	$(LAUNCHER) ./$(program) -i $(SLU_PATH)/nicslu/demo/ASIC_100k.mtx
