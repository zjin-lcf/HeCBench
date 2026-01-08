CC        = /opt/nvidia/hpc_sdk/Linux_aarch64/25.11/compilers/bin/nvc++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
PROFILE   = yes
LAUNCHER  =


# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker flags
LDFLAGS = 

ifeq ($(PROFILE),yes)
  CFLAGS += -DPROFILE_TMP
endif

# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-mp=gpu
else
  CFLAGS +=-mp=multicore
endif

SRC = streamcluster.cpp
EXE = streamcluster

$(EXE):$(SRC) kernel.h streamcluster.h  streamcluster_cl.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(LDFLAGS)

run: $(EXE)
	$(LAUNCHER) ./$(EXE) 10 20 256 65536 65536 1000 none output.txt 1 

clean: 
	rm -f $(EXE) output.txt result*
