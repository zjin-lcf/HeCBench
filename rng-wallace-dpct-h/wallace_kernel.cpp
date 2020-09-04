#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// ************************************************
// wallace_kernel.cu
// authors: Lee Howes and David B. Thomas
//
// Wallace random number generator demonstration code.
//
// Contains code both for simply generating the
// next number, such that can be called by
// the options simulations and also for generatedRandomNumberPoolting
// directly into a memory buffer.
//
// Note that in this code, unlike in the descriptions in the chapter, we have
// given each thread a pair of 4 element transforms to perform, each using a
// slightly different hadamard matrix. The reason behind this is that the
// complexity of computation caused a register shortage when 512 threads were
// needed (2048 pool/4) which is solved by doubling the number of values
// computed per thread and halving the number of threads.
// ************************************************

void Hadamard4x4a(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = p - t;
	q = q - t;
	r = t - r;
	s = t - s;
}

void Hadamard4x4b(float &p, float &q, float &r, float &s)
{
	float t = (p + q + r + s) / 2;
	p = t - p;
	q = t - q;
	r = r - t;
	s = s - t;
}

SYCL_EXTERNAL void rng_wallace(unsigned m_seed, float *globalPool,
                               float *generatedRandomNumberPool,
                               float *chi2Corrections,
                               sycl::nd_item<3> item_ct1, float *pool)
{

        const unsigned lcg_a = 241;
	const unsigned lcg_c = 59;
	const unsigned lcg_m = 256;
	const unsigned mod_mask = lcg_m - 1;

 unsigned offset =
     sycl::mul24((int)WALLACE_POOL_SIZE, (int)(item_ct1.get_group(2)));

#pragma unroll
	for (int i = 0; i < 8; i++)
  pool[item_ct1.get_local_id(2) + WALLACE_NUM_THREADS * i] =
      globalPool[offset + item_ct1.get_local_id(2) + WALLACE_NUM_THREADS * i];

 item_ct1.barrier();

        // Loop generating generatedRandomNumberPools repeatedly
	for (int loop = 0; loop < WALLACE_NUM_OUTPUTS_PER_RUN; loop++)
	{

		m_seed = (1664525U * m_seed + 1013904223U) & 0xFFFFFFFF;

  unsigned intermediate_address =
      sycl::mul24(loop, (int)(8 * WALLACE_TOTAL_NUM_THREADS)) +
      sycl::mul24((int)(8 * WALLACE_NUM_THREADS),
                  (int)(item_ct1.get_group(2))) +
      item_ct1.get_local_id(2);

  if (item_ct1.get_local_id(2) == 0)
   pool[WALLACE_CHI2_OFFSET] =
       chi2Corrections[sycl::mul24((int)(item_ct1.get_group(2)),
                                   (int)WALLACE_NUM_OUTPUTS_PER_RUN) +
                       loop];
  item_ct1.barrier();
                float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
		for (int i = 0; i < 8; i++)
		{
   generatedRandomNumberPool[intermediate_address + i * WALLACE_NUM_THREADS] =
       pool[sycl::mul24(i, (int)WALLACE_NUM_THREADS) +
            item_ct1.get_local_id(2)] *
       chi2CorrAndScale;
                }

		float rin0_0, rin1_0, rin2_0, rin3_0, rin0_1, rin1_1, rin2_1, rin3_1;
		for (int i = 0; i < WALLACE_NUM_POOL_PASSES; i++)
		{
   unsigned seed = (m_seed + item_ct1.get_local_id(2)) & mod_mask;
   item_ct1.barrier();
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin0_0 = pool[((seed << 3))];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin1_0 = pool[((seed << 3) + 1)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin2_0 = pool[((seed << 3) + 2)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin3_0 = pool[((seed << 3) + 3)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin0_1 = pool[((seed << 3) + 4)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin1_1 = pool[((seed << 3) + 5)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin2_1 = pool[((seed << 3) + 6)];
   seed = (sycl::mul24((int)seed, (int)lcg_a) + lcg_c) & mod_mask;
                        rin3_1 = pool[((seed << 3) + 7)];

   item_ct1.barrier();

                        Hadamard4x4a(rin0_0, rin1_0, rin2_0, rin3_0);
   pool[0 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin0_0;
   pool[1 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin1_0;
   pool[2 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin2_0;
   pool[3 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin3_0;

                        Hadamard4x4b(rin0_1, rin1_1, rin2_1, rin3_1);
   pool[4 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin0_1;
   pool[5 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin1_1;
   pool[6 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin2_1;
   pool[7 * WALLACE_NUM_THREADS + item_ct1.get_local_id(2)] = rin3_1;

   item_ct1.barrier();
                }
	}

 item_ct1.barrier();

#pragma unroll
	for (int i = 0; i < 8; i++)
  globalPool[offset + item_ct1.get_local_id(2) + WALLACE_NUM_THREADS * i] =
      pool[item_ct1.get_local_id(2) + WALLACE_NUM_THREADS * i];
}

