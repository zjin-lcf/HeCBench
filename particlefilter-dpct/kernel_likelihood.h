#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void kernel_likelihood(float *arrayX, float *arrayY, const float *xj,
                       const float *yj, int *ind, const int *objxy,
                       float *likelihood, const unsigned char *I,
                       float *weights, int *seed, float *partial_sums,
                       const int Nparticles, const int countOnes,
                       const int IszY, const int Nfr, const int k,
                       const int max_size, sycl::nd_item<3> item_ct1,
                       float *weights_local)
{

 int block_id = item_ct1.get_group(2);
 int thread_id = item_ct1.get_local_id(2);
 int i = block_id * item_ct1.get_local_range().get(2) + thread_id;
        int y;
	int indX, indY;
	float u, v;

	if(i < Nparticles){
		arrayX[i] = xj[i]; 
		arrayY[i] = yj[i]; 

		weights[i] = 1 / ((float) (Nparticles)); 
		seed[i] = (A*seed[i] + C) % M;
  u = sycl::fabs(seed[i] / ((float)M));
                seed[i] = (A*seed[i] + C) % M;
  v = sycl::fabs(seed[i] / ((float)M));
  arrayX[i] += 1.0 + 5.0 * (sycl::sqrt(-2 * sycl::log(u)) *
                            sycl::cos((float)(2 * PI * v)));

                seed[i] = (A*seed[i] + C) % M;
  u = sycl::fabs(seed[i] / ((float)M));
                seed[i] = (A*seed[i] + C) % M;
  v = sycl::fabs(seed[i] / ((float)M));
  arrayY[i] += -2.0 + 2.0 * (sycl::sqrt(-2 * sycl::log(u)) *
                             sycl::cos((float)(2 * PI * v)));
        }

 item_ct1.barrier();

        if(i < Nparticles)
	{
		for(y = 0; y < countOnes; y++){

			int iX = arrayX[i];
			int iY = arrayY[i];
			int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
			int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
			indX = rnd_iX + objxy[y*2 + 1];
			indY = rnd_iY + objxy[y*2];

   ind[i * countOnes + y] = sycl::abs(indX * IszY * Nfr + indY * Nfr + k);
                        if(ind[i*countOnes + y] >= max_size)
				ind[i*countOnes + y] = 0;
		}
		float likelihoodSum = 0.0;
		for(int x = 0; x < countOnes; x++)
			likelihoodSum += ((I[ind[i*countOnes + x]] - 100) * (I[ind[i*countOnes + x]] - 100) -
					(I[ind[i*countOnes + x]] - 228) * (I[ind[i*countOnes + x]] - 228)) / 50.0;
		likelihood[i] = likelihoodSum/countOnes-SCALE_FACTOR;

  weights[i] =
      weights[i] * sycl::exp(likelihood[i]); // Donnie Newell - added the
                                             // missing exponential function call
        }

	weights_local[thread_id] = 0.0; //weights_local[thread_id] = i;

 item_ct1.barrier();

        if(i < Nparticles){
		weights_local[thread_id] = weights[i];
	}

 item_ct1.barrier();

        for(unsigned int s=BLOCK_SIZE/2; s>0; s>>=1)
	{
		if(thread_id < s)
		{
			weights_local[thread_id] += weights_local[thread_id + s];
		}
  item_ct1.barrier();
        }
	if(thread_id == 0)
	{
		partial_sums[block_id] = weights_local[0];
	}
}
