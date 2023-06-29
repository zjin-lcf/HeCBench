/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sycl/sycl.hpp>

#include "KeccakF.h"
#include "KeccakTreeCPU.h"
#include "KeccakTreeGPU.h"

// choose 8 for fast execution
#define IMAX 8 // 1600 //2400 // 1600 for high speed mesures // iteration for speed mesure loops


//debug print function
void print_out(tKeccakLane * h_outBuffer,int nb_threads)
{
   printf("%08x ",h_outBuffer[0]);printf("%08x ",h_outBuffer[1]);
   printf("%08x ",h_outBuffer[nb_threads]);printf("%08x ",h_outBuffer[nb_threads +1]);
   printf("\n\n");
}

void TestCPU(int reduc)
{
   time_t t1,t2;
   double speed1;
   int i;

   tKeccakLane *h_inBuffer;// Host in buffer for data to be hashed
   tKeccakLane *h_outBuffer;// Host out buffer

   tKeccakLane Kstate[25]; //Keccak State for top node
   memset(Kstate, 0, 25 * sizeof(tKeccakLane));

   //init host inBuffer
   h_inBuffer=(tKeccakLane *) malloc( INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK );
   memset(h_inBuffer, 0, INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK);

   //init host outBuffer
   h_outBuffer=(tKeccakLane *) malloc( OUTPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS );
   memset(h_outBuffer, 0, OUTPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS );

   //***************************
   //init h_inBuffer with values
   for(i=0;i<INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK * NB_THREADS*NB_THREADS_BLOCKS;i++) h_inBuffer[i]=i;

   //CPU computation *******************************
   printf("CPU speed test started \n");

   t1=time(NULL);

   for(i=0;i<(IMAX/reduc);i++)
   {
      KeccakTreeCPU(h_inBuffer,h_outBuffer);

      //print_out(h_outBuffer,NB_THREADS);
      Keccak_top(Kstate,h_outBuffer,NB_THREADS*NB_THREADS_BLOCKS);
   }

   t2=time(NULL);

   print_KS_256(Kstate);

   speed1= (INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK *(IMAX/(reduc*1000.)))  / ((t2-t1) + 0.01);
   printf("CPU speed : %.2f kB/s \n\n",speed1);

   //free all buffer host and device
   free(h_inBuffer);
   free(h_outBuffer);
}

void TestGPU()
{
   time_t t1,t2;
   double speed1;
   unsigned int i;

   const tKeccakLane KeccakF_RoundConstants[22] =
   {
      (tKeccakLane)0x00000001 ,
      (tKeccakLane)0x00008082 ,
      (tKeccakLane)0x0000808a ,
      (tKeccakLane)0x80008000 ,
      (tKeccakLane)0x0000808b ,
      (tKeccakLane)0x80000001 ,
      (tKeccakLane)0x80008081 ,
      (tKeccakLane)0x00008009 ,
      (tKeccakLane)0x0000008a ,
      (tKeccakLane)0x00000088 ,
      (tKeccakLane)0x80008009 ,
      (tKeccakLane)0x8000000a ,
      (tKeccakLane)0x8000808b ,
      (tKeccakLane)0x0000008b ,
      (tKeccakLane)0x00008089 ,
      (tKeccakLane)0x00008003 ,
      (tKeccakLane)0x00008002 ,
      (tKeccakLane)0x00000080 ,
      (tKeccakLane)0x0000800a ,
      (tKeccakLane)0x8000000a ,
      (tKeccakLane)0x80008081 ,
      (tKeccakLane)0x00008080
   };

   tKeccakLane *h_inBuffer;// Host in buffer for data to be hashed
   tKeccakLane *h_outBuffer;// Host out buffer

   tKeccakLane Kstate[25]; //Keccak State for top node
   memset(Kstate, 0, 25 * sizeof(tKeccakLane));

   //init host inBuffer
   h_inBuffer=(tKeccakLane *) malloc( INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK );
   memset(h_inBuffer, 0, INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK);

   //init host outBuffer
   h_outBuffer=(tKeccakLane *) malloc( OUTPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS );
   memset(h_outBuffer, 0, OUTPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS );

   //***************************
   //init h_inBuffer with values
   for(i=0;i<INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK * NB_THREADS*NB_THREADS_BLOCKS;i++ )
   {h_inBuffer[i]=i;}

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

   tKeccakLane *d_inBuffer = sycl::malloc_device<tKeccakLane>(
     INPUT_BLOCK_SIZE_B/4 * NB_THREADS * NB_THREADS_BLOCKS * NB_INPUT_BLOCK, q);
   tKeccakLane *d_outBuffer = sycl::malloc_device<tKeccakLane>(
     OUTPUT_BLOCK_SIZE_B/4 * NB_THREADS * NB_THREADS_BLOCKS, q);
   tKeccakLane *d_KeccakF_RoundConstants = sycl::malloc_device<tKeccakLane>(22, q);
   q.memcpy(d_KeccakF_RoundConstants, KeccakF_RoundConstants, 22 * sizeof(tKeccakLane));

   //GPU computation *******************************
   printf("GPU speed test started\n");

   t1=time(NULL);

   for(i=0;i<IMAX;i++)
   {
      KeccakTreeGPU(q, h_inBuffer, d_inBuffer, h_outBuffer, d_outBuffer, d_KeccakF_RoundConstants);
      //print_out(h_outBuffer,NB_THREADS*NB_THREADS_BLOCKS);

      Keccak_top(Kstate,h_outBuffer,NB_THREADS*NB_THREADS_BLOCKS);
      //print_KS_256(Kstate);
   }

   t2=time(NULL);

   print_KS_256(Kstate);

   speed1= (INPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS * NB_INPUT_BLOCK *(IMAX/1000.))  / ((t2-t1) + 0.01);
   printf("GPU speed : %.2f kB/s \n\n",speed1);

   //free all buffer host and device
   free(h_inBuffer);
   free(h_outBuffer);
   sycl::free(d_inBuffer, q);
   sycl::free(d_outBuffer, q);
   sycl::free(d_KeccakF_RoundConstants, q);
}

void Print_Param(void)
{
   printf("\n");
   printf("Numbers of Threads PER BLOCK            NB_THREADS           %u \n", NB_THREADS);
   printf("Numbers of Threads Blocks               NB_THREADS_BLOCKS    %u \n", NB_THREADS_BLOCKS);
   printf("\n");
   printf("Input block size of Keccak (in Byte)    INPUT_BLOCK_SIZE_B   %u \n", INPUT_BLOCK_SIZE_B);
   printf("Output block size of Keccak (in Byte)   OUTPUT_BLOCK_SIZE_B  %u \n", OUTPUT_BLOCK_SIZE_B);
   printf("\n");
   printf("NB of input blocks in by Threads        NB_INPUT_BLOCK       %u \n", NB_INPUT_BLOCK );
   printf("\n");
}
