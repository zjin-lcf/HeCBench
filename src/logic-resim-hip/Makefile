HIP_ARCH ?= gfx906

.PHONY: all clean run

all:
	$(MAKE) -C Simulation HIP_ARCH=$(HIP_ARCH)

clean:
	$(MAKE) -C Simulation clean

run:
	$(MAKE) -C Simulation run
