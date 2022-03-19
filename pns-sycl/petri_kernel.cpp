/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifndef _PETRINET_KERNEL_H_
#define _PETRINET_KERNEL_H_

#include "petri.h"

#define BLOCK_SIZE 256
#define BLOCK_SIZE_BITS 8

void fire_transition(nd_item<1> &item,
                     char* g_places, int* conflict_array, int tr, 
		     int tc, int step, int N, int thd_thrd) 
{
  int val1, val2, val3, to_update;
  int mark1, mark2;
	
  to_update = 0;
  if (item.get_local_id(0)<thd_thrd) 
    {
      // check if the transition is enabled and conflict-free
      val1 = (tr==0)? (N+N)-1: tr-1;
      val2 = (tr & 0x1)? (tc==N-1? 0: tc+1): tc;
      val3 = (tr==(N+N)-1)? 0: tr+1;
      mark1 = g_places[val1*N+val2];
      mark2 = g_places[tr*N+tc];
      if ( (mark1>0) && (mark2>0) ) 
	{
	  to_update = 1;
	  conflict_array[tr*N+tc] = step;
	}
    }
  __syncthreads();

  if (to_update) 
    {
      // If there are conflicts, transitions on even/odd rows are 
      // kept when the step is even/odd
      to_update = ((step & 0x01) == (tr & 0x01) ) || 
	( (conflict_array[val1*N+val2]!=step) && 
	  (conflict_array[val3*N+((val2==0)? N-1: val2-1)]!=step) );
    }

  // now update state
  // 6 kernel memory accesses 
  if (to_update) 
    {
      g_places[val1*N+val2] = mark1-1;  // the place above
      g_places[tr*N+tc] = mark2-1; // the place on the left
    }
  __syncthreads();
  if (to_update) 
    {
      g_places[val3*N+val2]++;  // the place below
      g_places[tr*N+(tc==N-1? 0: tc+1)]++; // the place on the right
    }
  __syncthreads();
}


void initialize_grid(nd_item<1> &item, uint32* mt, int* g_places, int nsquare2, int seed) 
{
  // N is an even number
  int i;
  int loop_num = nsquare2 >> (BLOCK_SIZE_BITS+2);
  int threadIdx_x = item.get_local_id(0);
	
  for (i=0; i<loop_num; i++) 
    {
      g_places[threadIdx_x+(i<<BLOCK_SIZE_BITS)] = 0x01010101;
    }
    
  if (threadIdx_x < (nsquare2>>2)-(loop_num<<BLOCK_SIZE_BITS)) 
    {
      g_places[threadIdx_x+(loop_num<<BLOCK_SIZE_BITS)] = 0x01010101;
    }
	
  RandomInit(item, mt, item.get_group(0)+seed);
}

void run_trajectory(nd_item<1> &item, 
                    uint32* mt,
                    int* g_places, int N, int max_steps) 
{
  int step, NSQUARE2, val;

  step = 0;
  NSQUARE2 = (N+N)*N;
	
  int threadIdx_x = item.get_local_id(0);

  while (step<max_steps) 
    {
      BRandom(item, mt); // select the next MERS_N (624) transitions

      // process 256 transitions
      val = mt[threadIdx_x]%NSQUARE2;
      fire_transition(item, (char*)g_places, g_places+(NSQUARE2>>2), 
		      val/N, val%N, step+7, N, BLOCK_SIZE);
      
      // process 256 transitions
      val = mt[threadIdx_x+BLOCK_SIZE]%NSQUARE2;
      fire_transition(item, (char*)g_places, g_places+(NSQUARE2>>2), 
		      val/N, val%N, step+11, N, BLOCK_SIZE);
		                
      // process 112 transitions
      if (  threadIdx_x < MERS_N-(BLOCK_SIZE<<1)  ) 
	{
	  val = mt[threadIdx_x+(BLOCK_SIZE<<1)]%NSQUARE2;
	}
      fire_transition(item, (char*)g_places, g_places+(NSQUARE2>>2), 
		      val/N, val%N, step+13, N, MERS_N-(BLOCK_SIZE<<1));

      step += MERS_N>>1; 
      // experiments show that for N>2000 and max_step<20000, 
      // the step increase is larger than 320
    }
}


void compute_reward_stat(nd_item<1> &item,
                         uint32* __restrict mt,
                         int* __restrict g_places,
                         float* __restrict g_vars,
                         int* __restrict g_maxs, 
			 int NSQUARE2) 
{
  float sum = 0;
  int i;
  int max = 0;
  int temp, data; 
  int loop_num = NSQUARE2 >> (BLOCK_SIZE_BITS+2);
  int threadIdx_x = item.get_local_id(0);
  int blockIdx_x = item.get_group(0);

  for (i=0; i<=loop_num-1; i++) 
    {  // a bug. i<loop_num should be changed to i<=loop_num-1
      data = g_places[threadIdx_x+(i<<BLOCK_SIZE_BITS)];
	    
      temp = data & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>8) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>16) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>24) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
    }

  i = NSQUARE2>>2;
  i &= 0x0FF;
  loop_num *= BLOCK_SIZE; 
  // I do not know why loop_num<<=BLOCK_SIZE_BITS does not work
  if (threadIdx_x <= i-1) 
    {
      data = g_places[threadIdx_x+loop_num];
	    
      temp = data & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>8) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>16) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
      temp = (data>>24) & 0x0FF;
      sum += temp*temp;
      max = max<temp? temp: max;
    }
	
  ((float*)mt)[threadIdx_x] = (float)sum;
  mt[threadIdx_x+BLOCK_SIZE] = (uint32)max;
  __syncthreads();
		
  for (i=(BLOCK_SIZE>>1); i>0; i = (i>>1) ) 
    {
      if (threadIdx_x<i) 
	{
	  ((float*)mt)[threadIdx_x] += ((float*)mt)[threadIdx_x+i];
	  if (mt[threadIdx_x+BLOCK_SIZE]<mt[threadIdx_x+i+BLOCK_SIZE])
	    mt[threadIdx_x+BLOCK_SIZE] = mt[threadIdx_x+i+BLOCK_SIZE];
	}
      __syncthreads();
    }
		
  if (threadIdx_x==0) 
    {
      g_vars[blockIdx_x] = (((float*)mt)[0])/NSQUARE2-1; 
      // D(X)=E(X^2)-E(X)^2, E(X)=1
      g_maxs[blockIdx_x] = (int)mt[BLOCK_SIZE];
    }
}

// Kernel function for simulating Petri Net for a defined grid
// n: the grid has 2nX2n places and transitions together
// s: steps in each trajectory
// t: number of trajectories
void PetrinetKernel(
  nd_item<1> &item,
  uint32* __restrict mt,
  int* __restrict g_s,
  float* __restrict g_v,
  int* __restrict g_m,
  int n, int s, int seed) 
{
  // block size must be 256
  // n is an even number
  int nsquare2 = n*n*2;
  int* g_places = g_s + item.get_group(0) * ((nsquare2>>2)+nsquare2);   
  // place numbers, conflict_array
  initialize_grid(item, mt, g_places, nsquare2, seed);
  
  run_trajectory(item, mt, g_places, n, s);
  compute_reward_stat(item, mt, g_places, g_v, g_m, nsquare2);
}

#endif // #ifndef _PETRINET_KERNEL_H_
