#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = gcc
CXX       = g++
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_121
LAUNCHER  =

#===============================================================================
# CMake Build Configuration
#===============================================================================

BUILD_DIR = build
# Extract numeric value from ARCH (sm_121 -> 121) for CMake
ARCH_NUM = $(subst sm_,,$(ARCH))
CMAKE_OPTS = -DCMAKE_CUDA_ARCHITECTURES=$(ARCH_NUM)

ifeq ($(DEBUG),yes)
  CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Debug
else
  CMAKE_OPTS += -DCMAKE_BUILD_TYPE=Release
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run

all: build

build: $(BUILD_DIR)/Makefile
	$(MAKE) -C $(BUILD_DIR)

$(BUILD_DIR)/Makefile:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_OPTS) ..

clean:
	rm -rf $(BUILD_DIR)

run: build
	$(LAUNCHER) ./$(BUILD_DIR)/daphne

test: run
