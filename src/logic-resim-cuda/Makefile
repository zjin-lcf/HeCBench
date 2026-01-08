# Wrapper Makefile for logic-resim benchmark
# Delegates to Simulation/Makefile

.PHONY: all clean run

all:
	$(MAKE) -C Simulation ARCH=$(ARCH)

clean:
	$(MAKE) -C Simulation clean

run:
	@echo "Note: logic-resim requires intermediate files and test data."
	@echo "Please see README.md for usage instructions."
	@echo "Test data available in Simulation/toy/"
