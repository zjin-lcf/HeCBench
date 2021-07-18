CC = hipcc

# -DDEBUG sets a constant seed for random number generation
CFLAGS = -O3 -std=c++11 -ffast-math -DDEBUG


SOURCES = main.cu lsqt.cu sigma.cu vector.cu hamiltonian.cu model.\
	cu model_general.cu model_lattice.cu anderson.cu charge.cu
	
_OBJ = main.o lsqt.o sigma.o vector.o hamiltonian.o model.o model_general.o \
	model_lattice.o anderson.o charge.o

HEADERS = sigma.h vector.h lsqt.h hamiltonian.h model.h anderson.h charge.h common.h


ODIR = obj_gpu
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: lsqt_gpu

$(ODIR)/%.o: %.cu $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

lsqt_gpu: $(OBJ)
	$(CC) $^ -o $@

clean:
	rm -f $(ODIR)/*.o lsqt_gpu *.out

run: lsqt_gpu
	./lsqt_gpu ../lsqt-cuda/examples/input.txt
	
