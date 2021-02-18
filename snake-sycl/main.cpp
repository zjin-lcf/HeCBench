/*
 * Copyright (c) <2017 - 2020>, ETH Zurich and Bilkent University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or other
 *   materials provided with the distribution.
 * - Neither the names of the ETH Zurich, Bilkent University,
 *   nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without specific
 *   prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include "common.h"

using namespace std::chrono;

#define warp_size 32
#define SharedPartDevice 64
#define FULL_MASK 0xffffffff
#define NBytes Nuints
#define PRINT 0

#define Number_of_Diagonals 9
#define F_ReadLength 100
#define BitVal(data,y) ( (data>>y) & 1)    // Return Data.Y value
#define SetBit(data,y) data |= (1 << y)    // Set Data.Y   to 1


int main(int argc, const char * const argv[])
{
  if (argc != 4){
    printf("Incorrect arguments..\nUsage: ./%s [ReadLength] [ReadandRefFile] [#reads]\n", argv[0]);
    exit(-1);
  }

  int ReadLength = atoi(argv[1]);//in my inputs, it is always 100. Just for the generality we keep it as a variable
  int NumReads = atoi(argv[3]); // Number of reads
  int Size_of_uint_in_Bit = 32; //in Bits 

  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  char *p;//when reading each char_basepair from the file, we read it into the p.

  int Number_of_warps_inside_each_block = 8; 
  int Concurrent_threads_In_Block = warp_size * Number_of_warps_inside_each_block;
  int Number_of_blocks_inside_each_kernel = (NumReads + Concurrent_threads_In_Block - 1) / 
                                            Concurrent_threads_In_Block;

  int F_ErrorThreshold =0;

  uint* ReadSeq = (uint * ) calloc(NumReads * 8, sizeof(uint));
  uint* RefSeq = (uint * ) calloc(NumReads * 8, sizeof(uint));
  int* DFinal_Results = (int * ) calloc(NumReads, sizeof(int));


  int tokenIndex=1;
  fp = fopen(argv[2], "r");
  if (!fp){
    printf("Sorry, the file does not exist or you do not have access permission\n");
    return 0;
  }
  for(int this_read = 0; this_read < NumReads; this_read++) {
    read = getline(&line, &len, fp);
    tokenIndex=1;
    for (p = strtok(line, "\t"); p != NULL; p = strtok(NULL, "\t"))
    {
      if (tokenIndex==1)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            //do nothing (this is like storing 00)
          }
          else if (p[j] == 'C')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));

            ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          //printf("%c",p[j]);
          //printf(" %08x", ReadSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)]);
        }
      }
      else if(tokenIndex==2)
      {
        for (int j = 0; j < ReadLength; j++)
        {
          if(p[j] == 'A')
          {
            //do nothing (this is like storing 00)
          }
          else if (p[j] == 'C')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          else if (p[j] == 'G')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));
          }
          else if (p[j] == 'T')
          {
            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2));

            RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)] = SetBit(RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)], 31 - ((j%(Size_of_uint_in_Bit/2)) * 2 + 1));
          }
          //printf("%c",p[j]);
          //printf(" %08x", RefSeq[((j*2/Size_of_uint_in_Bit) + this_read * NBytes)]);
        }
      }
      tokenIndex=tokenIndex+1;
    }
  }
  fclose(fp);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<uint, 1> Dev_ReadSeq (NumReads * 8);
  buffer<uint, 1> Dev_RefSeq (NumReads * 8);
  buffer<int, 1> Dtest_Results (NumReads);

  range<1> gws(Concurrent_threads_In_Block * Number_of_blocks_inside_each_kernel);
  range<1> lws(Concurrent_threads_In_Block);

  q.submit([&] (handler &cgh) {
    auto acc = Dev_ReadSeq.get_access<sycl_write>(cgh);
    cgh.copy(ReadSeq, acc);
  });
  q.submit([&] (handler &cgh) {
    auto acc = Dev_RefSeq.get_access<sycl_write>(cgh);
    cgh.copy(RefSeq, acc);
  });

  for (int n = 0; n < 100; n++) {
    for (int loopPar = 0; loopPar <= 25; loopPar++) {
      F_ErrorThreshold = (loopPar*ReadLength)/100;

      high_resolution_clock::time_point t1 = high_resolution_clock::now();

      q.submit([&] (handler &cgh) {
        auto F_ReadSeq = Dev_ReadSeq.get_access<sycl_read>(cgh); 
        auto F_RefSeq = Dev_RefSeq.get_access<sycl_read>(cgh); 
        auto Ftest_Results = Dtest_Results.get_access<sycl_write>(cgh);
        cgh.parallel_for<class sneaky_snake>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int tid = item.get_global_id(0); 
          if(tid >= NumReads) return;

          // const int NBytes = 8;
          uint ReadsPerThread[NBytes];
          uint RefsPerThread[NBytes];

          #pragma unroll
          for (int i = 0; i < NBytes; i++)
          {
            ReadsPerThread[i] = F_ReadSeq[tid*8 + i];
            RefsPerThread[i] = F_RefSeq[tid*8 + i];
          }

          /////////////////////////////////////////////////////////////////////////////
          Ftest_Results[tid] = 1;

          uint ReadCompTmp = 0;
          uint RefCompTmp = 0;
          uint DiagonalResult = 0;

          uint ReadTmp1 = 0;
          uint ReadTmp2 = 0;

          uint RefTmp1 = 0;
          uint RefTmp2 = 0;

          uint CornerCase = 0;

          int localCounter= 0;
          int localCounterMax=0;
          int globalCounter = 0;
          int Max_leading_zeros = 0;
          int AccumulatedErrs = 0;

          int ShiftValue = 0;
          int Diagonal = 0;

          int j = 0; //specifying the j-th uint that we are reading in each read-ref comparison (can be from 0 to 7)

          while ( (j < 7) && (globalCounter < 200))
          {
            Diagonal = 0;
            RefTmp1 = RefsPerThread[j] << ShiftValue;
            RefTmp2 = RefsPerThread[j + 1] >>  32 - ShiftValue;
            ReadTmp1 = ReadsPerThread[j] << ShiftValue;
            ReadTmp2 = ReadsPerThread[j + 1] >>  32 - ShiftValue;

            ReadCompTmp = ReadTmp1 | ReadTmp2;
            RefCompTmp = RefTmp1 | RefTmp2;
            DiagonalResult = ReadCompTmp ^ RefCompTmp;
            localCounterMax = cl::sycl::clz(DiagonalResult);

            //////////////////// Upper diagonals /////////////////////

            for(int e = 1; e <= F_ErrorThreshold; e++)
            {
              Diagonal += 1;
              CornerCase = 0;
              if (  (j == 0)  &&  (  (ShiftValue - (2*e))  < 0 )  )
              {
                ReadTmp1 = ReadsPerThread[j] >> ( (2*e) - ShiftValue );
                ReadTmp2 = 0;

                ReadCompTmp = ReadTmp1 | ReadTmp2;
                RefCompTmp = RefTmp1 | RefTmp2;

                DiagonalResult = ReadCompTmp ^ RefCompTmp;

                CornerCase = 0;
                for(int Ci = 0; Ci < (2*e) - ShiftValue; Ci++)
                {
                  SetBit(CornerCase, 31 - Ci);
                }

                DiagonalResult  = DiagonalResult | CornerCase;
                localCounter = cl::sycl::clz(DiagonalResult);

              }
              else if ( (ShiftValue - (2*e) ) < 0 )
              {
                ReadTmp1 = ReadsPerThread[j-1] << 32 - ( (2*e) - ShiftValue );
                ReadTmp2 = ReadsPerThread[j] >> (2*e) - ShiftValue;

                ReadCompTmp = ReadTmp1 | ReadTmp2;
                RefCompTmp = RefTmp1 | RefTmp2;

                DiagonalResult = ReadCompTmp ^ RefCompTmp;

                localCounter = cl::sycl::clz(DiagonalResult);
              }
              else
              {
                ReadTmp1 = ReadsPerThread[j] <<  ShiftValue - (2*e);
                ReadTmp2 = ReadsPerThread[j+1] >> 32 - (ShiftValue - (2*e) ) ;

                ReadCompTmp = ReadTmp1 | ReadTmp2;
                RefCompTmp = RefTmp1 | RefTmp2;

                DiagonalResult = ReadCompTmp ^ RefCompTmp;

                localCounter = cl::sycl::clz(DiagonalResult);

              }
              if (localCounter>localCounterMax)
                localCounterMax=localCounter;
            }


            /*
               sh = shift
               up = upper diagonal
               RC = ReadCompTmp
               FC = RefCompTmp
               D = DiagonalResult
               DN = diagonal
               LC = localCounter
             */

            //////////////////// Lower diagonals /////////////////////

            for(int e = 1; e <= F_ErrorThreshold; e++)
            {
              Diagonal += 1;
              CornerCase = 0;
              if ( j<5)//  ( (globalCounter + ShiftValue + (2*e) + 32) < 200) )
              {
                if ( (ShiftValue + (2*e) )  < 32)
                {
                  ReadTmp1 = ReadsPerThread[j] << ShiftValue + (2*e);
                  ReadTmp2 = ReadsPerThread[j+1] >> 32 - ( ShiftValue + (2*e) );

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp = RefTmp1 | RefTmp2;

                  DiagonalResult = ReadCompTmp ^ RefCompTmp;
                  localCounter = cl::sycl::clz(DiagonalResult);

                }
                else
                {
                  ReadTmp1 = ReadsPerThread[j+1] << ( ShiftValue + (2*e) ) % 32;
                  ReadTmp2 = ReadsPerThread[j+2] >>  32 - ( ( ShiftValue + (2*e) ) % 32 );

                  ReadCompTmp = ReadTmp1 | ReadTmp2;
                  RefCompTmp = RefTmp1 | RefTmp2;

                  DiagonalResult = 0xffffffff;//ReadCompTmp ^ RefCompTmp;

                  DiagonalResult = ReadCompTmp ^ RefCompTmp;

                  localCounter = cl::sycl::clz(DiagonalResult);
                }
              }
              else
              {
                //printf("HI3");
                ReadTmp1 = ReadsPerThread[j] << ShiftValue + (2*e);
                ReadTmp2 = ReadsPerThread[j+1] >>   32 - ( ShiftValue + (2*e) );

                ReadCompTmp = ReadTmp1 | ReadTmp2;
                RefCompTmp = RefTmp1 | RefTmp2;
                DiagonalResult = ReadCompTmp ^ RefCompTmp;

                CornerCase = 0;
                if ((globalCounter+32)>200 ) {

                  for(int Ci = ((globalCounter+32)-200); Ci < (((globalCounter+32)-200)+ 2*e); Ci++)
                  {
                    SetBit(CornerCase, Ci);
                  }
                }

                else if ((globalCounter+32)>=(200- (2*e))){

                  for(int Ci = 0; Ci < (2*e); Ci++)
                  {
                    SetBit(CornerCase, Ci);
                  }
                }
                DiagonalResult = DiagonalResult | CornerCase;

                localCounter = cl::sycl::clz(DiagonalResult);
              }

              if (localCounter>localCounterMax)
                localCounterMax=localCounter;

            }

            /*
               CC = CornerCase
               sh = shift
               up = upper diagonal
               RC = ReadCompTmp
               FC = RefCompTmp
               D = DiagonalResult
               DN = diagonal
               LC = localCounter
             */

            Max_leading_zeros = 0;

            if ( (j == 6) && ( ((localCounterMax/2)*2) >= 8)  )
            {
              Max_leading_zeros = 8;
              break;
            }
            else if( ((localCounterMax/2)*2) > Max_leading_zeros)
            {
              Max_leading_zeros = ((localCounterMax/2)*2);
            }

            if ( ( (Max_leading_zeros/2) < 16) && (j < 5) )
            {
              AccumulatedErrs += 1;
            }
            else if (  (j == 6) && ( (Max_leading_zeros/2) < 4) )
            {
              AccumulatedErrs += 1;
            }

            if(AccumulatedErrs > F_ErrorThreshold)
            {
              Ftest_Results[tid] = 0;
              break;
            }


            if(ShiftValue + Max_leading_zeros + 2 >= 32)
            {
              j += 1;
            }

            // ShiftValue_2Ref = (ShiftValue_2Ref + Max_leading_zeros + 2) %32;
            if (Max_leading_zeros == 32)
            {
              globalCounter += Max_leading_zeros;
            }
            else
            {
              ShiftValue = ((ShiftValue + Max_leading_zeros + 2) % 32);
              globalCounter += (Max_leading_zeros + 2);
            }
          }
       });
      });

      q.submit([&] (handler &cgh) {
        auto acc = Dtest_Results.get_access<sycl_read>(cgh);
        cgh.copy(acc, DFinal_Results);
      });
      q.wait();

      high_resolution_clock::time_point t2 = high_resolution_clock::now();

      double elapsed_time = duration_cast<microseconds>(t2 - t1).count();
      int accepted = 0;
      for(int i = 0; i < NumReads; i++)
      {
        if(DFinal_Results[i] == 1)
          accepted += 1;
      }

      printf("E: \t %d \t Snake-on-GPU: \t %5.4f \t Accepted: \t %10d \t Rejected: \t %10d\n", 
          F_ErrorThreshold, elapsed_time, accepted, NumReads - accepted);
    }
  }

  free(ReadSeq);
  free(RefSeq);
  free(DFinal_Results);
  return 0;
}
