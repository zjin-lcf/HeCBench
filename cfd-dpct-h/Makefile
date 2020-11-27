VENDOR      = intel
OPTIMIZE    = yes
DEBUG       = no
DUMP        = yes


# Standard flags
CFLAGS := -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Codeplay Compiler
ifneq (,$(findstring codeplay,$(VENDOR)))
  CC = compute++
  CFLAGS += -no-serial-memop -sycl -sycl-driver
  LDFLAGS += -lComputeCpp
endif

# Intel DPC++ Compiler
ifneq (,$(findstring intel,$(VENDOR)))
  CC = dpcpp
  CFLAGS += 
  LDFLAGS += 
endif

# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


ifeq ($(DUMP),yes)
  CFLAGS += -DOUTPUT
endif

SRC = euler3d.cpp
EXE = euler3d

$(EXE):$(SRC) 
	$(CC) $(CFLAGS) $(KERNEL_DIM) $(SRC) -o $(EXE) $(LDFLAGS)

run: $(EXE)
	./euler3d ../data/cfd/fvcorr.domn.097K 
	#./euler3d ../data/cfd/fvcorr.domn.193K   
	#./euler3d ../data/cfd/missile.domn.0.2M  


clean: 
	rm -f $(EXE) $(EXE).linkinfo result*  density momentum density_energy
