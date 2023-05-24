/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREEGPU_H_INCLUDED
#define KECCAKTREEGPU_H_INCLUDED

#include <sycl/sycl.hpp>
#include "KeccakTree.h"
#include "KeccakTypes.h"
#include "KeccakF.h"

//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************

void KeccakTreeGPU(sycl::queue &q,
                   tKeccakLane *h_inBuffer,
                   tKeccakLane *d_inBuffer,
                   tKeccakLane *h_outBuffer,
                   tKeccakLane *d_outBuffer,
                   tKeccakLane *d_KeccakF_RoundConstants);



#endif // KECCAKTREEGPU_H_INCLUDED
