/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREEGPU_H_INCLUDED
#define KECCAKTREEGPU_H_INCLUDED

#include <omp.h>
#include "KeccakTree.h"
#include "KeccakTypes.h"
#include "KeccakF.h"

//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************

#pragma omp declare target 
void KeccakTreeGPU(tKeccakLane * h_inBuffer, tKeccakLane * h_outBuffer,  const tKeccakLane *h_KeccakF_RoundConstants);
#pragma omp end declare target 



#endif // KECCAKTREEGPU_H_INCLUDED
