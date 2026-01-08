# Wrapper Makefile for local-ht benchmark
# Delegates to src/Makefile

.PHONY: all clean run

all:
	$(MAKE) -C src ARCH=$(ARCH)

clean:
	$(MAKE) -C src clean

run:
	$(MAKE) -C src run ARCH=$(ARCH)
