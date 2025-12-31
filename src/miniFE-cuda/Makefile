#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
ARCH      = sm_121
LAUNCHER  =

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run

all: build

build:
	$(MAKE) -C src ARCH=$(ARCH)

clean:
	$(MAKE) -C src clean

run: build
	cd src && $(LAUNCHER) ./miniFE.x

test: run
