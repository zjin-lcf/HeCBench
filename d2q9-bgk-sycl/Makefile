include ../makefile_sycl_include.mk
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	./$(program) Inputs/input_256x256.params Obstacles/obstacles_256x256.dat

check: $(REF_FINAL_STATE_FILE)  $(FINAL_STATE_FILE)
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) \
        --ref-final-state-file=$(REF_FINAL_STATE_FILE) \
        --av-vels-file=$(AV_VELS_FILE) \
        --final-state-file=$(FINAL_STATE_FILE)
