/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREEGPU_H_INCLUDED
#define KECCAKTREEGPU_H_INCLUDED

#include "KeccakTree.h"
#include "KeccakTypes.h"
#include "KeccakF.h"
#include "common.h"

//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************

void KeccakTreeGPU(queue &q, tKeccakLane * h_inBuffer, buffer<tKeccakLane,1> &d_inBuffer,
                   tKeccakLane * h_outBuffer, buffer<tKeccakLane,1> &d_outBuffer,
		   buffer<tKeccakLane,1> &d_KeccakF_RoundConstant);



#endif // KECCAKTREEGPU_H_INCLUDED
