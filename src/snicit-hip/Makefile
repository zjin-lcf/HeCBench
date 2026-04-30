HIP_ARCH ?= gfx906

.PHONY: all clean run

all:
	$(MAKE) -C bin HIP_ARCH=$(HIP_ARCH)

clean:
	$(MAKE) -C bin clean

run:
	$(MAKE) -C bin run
