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


#===============================================================================
# Program name & source code list
#===============================================================================

program = multimat multimat_FL multimat_F

all: $(program)

source=compact.cpp full_matrix.cpp multimat.cpp

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -Wall -I../include \
          --hipsycl-platform=$(PLATFORM) \
	  --hipsycl-gpu-arch=$(MARCH)

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

multimat: $(source)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
multimat_F: $(source)
	$(CC) $(CFLAGS) -DFUSED $^ -o $@ $(LDFLAGS)
multimat_FL: $(source)
	$(CC) $(CFLAGS) -DFUSED -DLINKED $^ -o $@ $(LDFLAGS)

clean: 
	rm -rf $(program)

run: $(program)
	./multimat_FL 2000 2000
	./multimat_F 2000 2000
	./multimat 2000 2000
	./multimat_FL 2000 2000 0.3 0.05 0.05
	./multimat_F 2000 2000 0.3 0.05 0.05
	./multimat 2000 2000 0.3 0.05 0.05


