#!/bin/bash

# Using OpenCL backend
export SYCL_BE=PI_OPENCL

# The Rodinia benchmarks
rodinia=(
b+tree-sycl
backprop-sycl
bfs-sycl
cfd-sycl
dwt2d-sycl
gaussian-sycl
heartwall-sycl
hotspot-sycl
hotspot3D-sycl
hybridsort-sycl
kmeans-sycl
lavaMD-sycl
leukocyte-sycl
lud-sycl
myocyte-sycl
nn-sycl
nw-sycl
particlefilter-sycl
pathfinder-sycl
srad-sycl
streamcluster-sycl )

for dir in "${rodinia[@]}"
do
  cd "${dir}"

  # DPC++
  make clean
  rm -f dpcpp_*.txt
  echo "====== ${dir} ======:"
  for (( i = 0; i < 4; i = i + 1 ))
  do
    cliloader -q -h -d make CC=dpcpp run &> dpcpp_gpu_report${i}.txt
  done
  grep -H "Total Time" dpcpp_gpu_report*.txt &> dpcpp_gpu_summary.txt

  make clean
  for (( i = 0; i < 4; i = i + 1 ))
  do
    cliloader -q -h -d make CC=dpcpp GPU=no run &> dpcpp_cpu_report${i}.txt
  done
  grep -H "Total Time" dpcpp_cpu_report*.txt &> dpcpp_cpu_summary.txt
  
  # ComputeCpp
  make clean
  rm -f cpcpp_*.txt 
  for (( i = 0; i < 4; i = i + 1 ))
  do
    cliloader -q -h -d make VENDOR=codeplay run &> cpcpp_gpu_report${i}.txt
  done
  grep -H "Total Time" cpcpp_gpu_report*.txt &> cpcpp_gpu_summary.txt

  make clean
  for (( i = 0; i < 4; i = i + 1 ))
  do
    cliloader -q -h -d make VENDOR=codeplay GPU=no run &> cpcpp_cpu_report${i}.txt
  done
  grep -H "Total Time" cpcpp_cpu_report*.txt &> cpcpp_cpu_summary.txt
  cd ..
done
