include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================
SLU_PATH = ../slu-cuda/src

SLU_LIBS = $(SLU_PATH)/nicslu/lib/nicslu.a \
           $(SLU_PATH)/nicslu/util/nicslu_util.a

program = main

obj = preprocess.o Timer.o numeric.o symbolic.o main.o $(SLU_LIBS)

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
	./$(program) -i $(SLU_PATH)/nicslu/test/add32.mtx

run: $(program)
	./$(program) -i $(SLU_PATH)/nicslu/demo/ASIC_100k.mtx
