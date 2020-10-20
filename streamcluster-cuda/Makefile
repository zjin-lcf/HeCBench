CC          = nvcc
OPTIMIZE    = yes
DEBUG       = no



# Standard flags
CFLAGS := -std=c++14 -Xcompiler -Wall

# Linker flags
LDFLAGS = -lm


# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif


SRC = streamcluster.cu
EXE = streamcluster

$(EXE):$(SRC) kernel.h streamcluster.h  streamcluster_cl.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(LDFLAGS)	

run: $(EXE)
	./$(EXE) 10 20 256 65536 65536 1000 none output.txt 1 

clean: 
	rm -f $(EXE) output.txt result*
