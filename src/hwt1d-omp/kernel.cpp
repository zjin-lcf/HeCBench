/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 */


/**
 * @brief   Calculates decomposed signal with maximum of 9 levels by using 
 *          1D Haar wavelet decomposition 
 * @param   inSignal        input signal
 * @param   coefsSignal     Coefficient details of signal after 9 levels of decompostion 
 * @param   AverageSignal   Averages of signal after 9 levels of decompostion
 * @param   sharedArray     shared array 
 * @param   tLevels         actual levels required for full decomposition 
 * @param   signalLength    length of signal
 * @param   levelsDone      level of decompositions done 
 * @param   mLevels	    maximum number of levels to be processed on device
 */

#include "hwt.h"

// begin of dwtHaar1D
void dwtHaar1D(const int numTeams,
               const int numThreads, 
               const float *__restrict__ inSignal,
                      float *__restrict__ coefsSignal,
                      float *__restrict__ AverageSignal,
                      const unsigned int tLevels,
                      const unsigned int signalLength,
                      const unsigned int levelsDone,
		      const unsigned int mLevels)
{
  #pragma omp target teams num_teams(numTeams) thread_limit(numThreads)
  {
    float sharedArray [512];
    #pragma omp parallel num_threads(numThreads)
    {
      size_t localId = omp_get_thread_num();
      size_t groupId = omp_get_team_num();
      size_t localSize = omp_get_num_threads();
      
      /**
       * Read input signal data from global memory
       * to shared memory
       */
      float t0 = inSignal[groupId * localSize * 2 + localId];
      float t1 = inSignal[groupId * localSize * 2 + localSize + localId];
      // Divide with signal length for normalized decomposition
      if(0 == levelsDone)
      {
         float r = 1.f / sqrtf((float)signalLength);
         t0 *= r;
         t1 *= r;
      }
      sharedArray[localId] = t0;
      sharedArray[localSize + localId] = t1;
       
      #pragma omp barrier
      
      unsigned int levels = tLevels > mLevels ? mLevels: tLevels;
      unsigned int activeThreads = (1 << levels) / 2;
      unsigned int midOutPos = signalLength / 2;
      
      const float rsqrt_two = 0.7071f;
      for(unsigned int i = 0; i < levels; ++i)
      {

          float data0, data1;
          if(localId < activeThreads)
          {
              data0 = sharedArray[2 * localId];
              data1 = sharedArray[2 * localId + 1];
          }

          /* make sure all work items have read from sharedArray before modifying it */
          #pragma omp barrier

          if(localId < activeThreads)
          {
              sharedArray[localId] = (data0 + data1) * rsqrt_two;
              unsigned int globalPos = midOutPos + groupId * activeThreads + localId;
              coefsSignal[globalPos] = (data0 - data1) * rsqrt_two;
         
              midOutPos >>= 1;
          }
          activeThreads >>= 1;
          #pragma omp barrier
      }
  
      /**
       * Write 0th element for the next decomposition
       * steps which are performed on host 
       */
      
       if(0 == localId)
          AverageSignal[groupId] = sharedArray[0];
    }
  }
}
// end of dwtHaar1D

