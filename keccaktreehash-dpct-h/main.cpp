/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "KeccakTreeCPU.h"
#include "KeccakTreeGPU.h"
#include "Test.h"


int main()
{
   Print_Param();
   TestCPU(1);
   TestGPU();
   return 0;
}
