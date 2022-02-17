#!/bin/bash

#ComputeCPP installed from https://developer.codeplay.com/products/computecpp/ce/download
export Oclgrind_DIR=/home/9bj/.oclgrind
export ComputeCpp_DIR=/home/9bj/computecpp/ComputeCpp-CE-2.7.0-x86_64-linux-gnu
export CPLUS_INCLUDE_PATH=/opt/rocm/opencl/include
export VENDOR=codeplay
export LD_LIBRARY_PATH=$ComputeCpp_DIR/lib:$Oclgrind_DIR/lib:$LD_LIBRARY_PATH
export PATH=$ComputeCpp_DIR/bin:$PATH
#for verbose logging with ComputeCpp:
echo "verbose_output=true" > computecpp.conf
export NUM_PASSES=1
#DPC++ installed with:
# git clone https://github.com/intel/llvm -b sycl
# python $HOME/Downloads/llvm/buildbot/configure.py -o build_dpc++_opencl
# python $HOME/Downloads/llvm/buildbot/compile.py -o build_dpc++_opencl
#export LD_LIBRARY_PATH=/home/9bj/Downloads/build_dpc++_opencl/lib:$LD_LIBRARY_PATH
#export PATH=/home/9bj/Downloads/build_dpc++_opencl/bin:$PATH
# For SYCL
#export SYCL_BE=PI_OPENCL

for dir in $(find . -mindepth 1 -maxdepth 1 -type d | grep -Ev '.\.git|include|cuda|omp|hip')
do
  cd "${dir}"
  echo "Benchmark: ${dir} Passes: ${NUM_PASSES}"
  rm -f aiwc_*.csv report*.txt
  make clean
  make VENDOR=$VENDOR GPU=no
  OCL_ICD_VENDORS=/home/9bj/.oclgrind/lib OCL_ICD_FILENAMES=liboclgrind-rt-icd.so OCLGRIND_WORKLOAD_CHARACTERISATION=1 COMPUTECPP_CONFIGURATION_FILE=computecpp.conf make run
  cd ..
done

