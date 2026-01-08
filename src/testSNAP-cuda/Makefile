#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -I. -I../testSNAP-omp \
          -DREFDATA_TWOJ=14  -arch=$(ARCH)

# Reference data files
REFDATA_FILES = refdata_2J14_W.h refdata_2J8_W.h refdata_2J4_W.h refdata_2J2_W.h

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

$(REFDATA_FILES):
	@if [ ! -f refdata_2J14_W.h ]; then \
		echo "Extracting SNAP reference data..."; \
		tar -xzf ../testSNAP-omp/refdata.tar.gz; \
	fi

%.o: %.cu snap.h utils.cu $(REFDATA_FILES)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nsteps 100
