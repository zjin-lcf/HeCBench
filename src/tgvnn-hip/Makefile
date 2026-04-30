HIP_ARCH ?= gfx906

.PHONY: all clean run

all:
	$(MAKE) -C src HIP_ARCH=$(HIP_ARCH)

clean:
	$(MAKE) -C src clean

run:
	$(MAKE) -C src run
