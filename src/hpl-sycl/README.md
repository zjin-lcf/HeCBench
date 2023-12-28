This is a workload for high performance linpack. <br />

## Dependencies
The OneAPI toolkit <br />

source /opt/intel/oneapi/setvars.sh <br />

## Build and run the benchmark <br />
cd src/hpl-2.3/ <br />
make clean && make <br />
cd bin/intel64/ <br />
cp ../../../../datafiles/HPL_small_gpu.dat HPL.dat <br />
export LD_LIBRARY_PATH=../../src/dpcpp/:$LD_LIBRARY_PATH <br />
mpirun -n 1 ./xhpl <br />


## view output <br />
### look for the GFlops measurement in the output log<br />
================================================================================ <br />
T/V                N    NB     P     Q               Time                 Gflops <br />
-------------------------------------------------------------------------------- <br />
WR10L2L2        4096   768     1     1               0.33              1.387e+02 <br />
-------------------------------------------------------------------------------- <br />
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=        0.0056536 ...... PASSED <br />
================================================================================ <br />

Finished      1 tests with the following results: <br />
              1 tests completed and passed residual checks, <br />
              0 tests completed and failed residual checks, <br />
              0 tests skipped because of illegal input values. <br />
-------------------------------------------------------------------------------- <br />


