.PHONY: all clean run

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

# src/Makefile builds miniFE.x but doesn't define a `run:` target; invoke
# the binary directly with a small default problem size.
run: all
	cd src && ./main -nx 128 -ny 128 -nz 128
