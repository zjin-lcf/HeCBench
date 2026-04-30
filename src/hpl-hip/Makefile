# HPL uses lowercase `arch` for CPU platform (intel64), so we use HIP_ARCH
# for the GPU architecture and pass it through to the inner build.
HIP_ARCH ?= gfx906

.PHONY: all clean

all:
	$(MAKE) -C src/hpl-2.3 HIP_ARCH=$(HIP_ARCH)

clean:
	$(MAKE) -C src/hpl-2.3 clean
