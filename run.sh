#!/bin/bash

# For SYCL
export SYCL_BE=PI_OPENCL

for dir in $(find . -mindepth 1 -maxdepth 1 -type d | grep -Ev '.\.git|include|cuda|omp|hip')
do
  cd "${dir}"
  make clean
  echo "${dir} results:"
  for (( i = 0; i < 10; i = i + 1 ))
  do
    cliloader -q -h -d make run &> report${i}.txt
  done
  grep -H "Total" report*.txt
  make clean
  cd ..
done

# For OMP target offloading
export LIBOMPTARGET_PLUGIN=OPENCL

for dir in $(find . -mindepth 1 -maxdepth 1 -type d | grep -Ev '.\.git|include|cuda|sycl|dpct|hip')
do
  cd "${dir}"
  make clean
  echo "${dir} results:"
  for (( i = 0; i < 10; i = i + 1 ))
  do
    cliloader -q -h -d make run &> report${i}.txt
  done
  grep -H "Total" report*.txt
  make clean
  cd ..
done
