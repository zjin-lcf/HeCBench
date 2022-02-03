include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = multimat multimat_FL multimat_F

source=compact.cpp full_matrix.cpp multimat.cpp

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
