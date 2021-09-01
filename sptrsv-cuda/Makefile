#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o sptrsv_syncfree.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++14 -Xcompiler -Wall -arch=$(ARCH) \
          -DVALUE_TYPE=double -I../sptrsv-sycl

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../sptrsv-sycl/main.cpp \
	../sptrsv-sycl/mmio.h \
	../sptrsv-sycl/read_mtx.h \
	../sptrsv-sycl/sptrsv.h \
	../sptrsv-sycl/tranpose.h
	$(CC) $(CFLAGS) -c $< -o $@

sptrsv_syncfree.o : sptrsv_syncfree.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) ../sptrsv-sycl/lp1.mtx

