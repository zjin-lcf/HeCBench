# si-hip is the only HIP benchmark that builds via CMake instead of Make.
# This wrapper drives the CMake build out-of-tree under release/ and
# matches the all/clean interface the rest of HeCBench expects.

.PHONY: all clean

all:
	mkdir -p release
	cd release && cmake $(CURDIR) -DCMAKE_BUILD_TYPE=Release
	$(MAKE) -C release

clean:
	rm -rf release
