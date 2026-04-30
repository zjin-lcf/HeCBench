HIP_ARCH ?= gfx906

.PHONY: all clean run

all:
	$(MAKE) -C src/points2image HIP_ARCH=$(HIP_ARCH)

clean:
	$(MAKE) -C src/points2image clean

run:
	$(MAKE) -C src/points2image run
