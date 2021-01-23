#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void kernel_normalize_weights(float *weights, const float *partial_sums,
                              float *CDF, float *u, int *seed,
                              const int Nparticles, sycl::nd_item<3> item_ct1,
                              float *u1, float *sumWeights)
{

 int local_id = item_ct1.get_local_id(2);
 int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
         item_ct1.get_local_id(2);
        if(0 == local_id)
  *sumWeights = partial_sums[0];
 item_ct1.barrier();
        if(i < Nparticles) {
 weights[i] = weights[i]/(*sumWeights);
	}
	item_ct1.barrier();
	if(i == 0) {
		CDF[0] = weights[0];
		for(int x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}

		seed[i] = (A*seed[i] + C) % M;
		float p = sycl::fabs(seed[i]/((float)M));
		seed[i] = (A*seed[i] + C) % M;
		float q = sycl::fabs(seed[i]/((float)M));
		u[0] = (1.0f/((float)(Nparticles))) * 
			(sycl::sqrt(-2.0f*sycl::log(p))*sycl::cos(2.0f*PI*q));
		// do this to allow all threads in all blocks to use the same u1
	}
	item_ct1.barrier();
	if(0 == local_id)
		*u1 = u[0];

	item_ct1.barrier();
	if(i < Nparticles)
	{
		u[i] = *u1 + i/((float)(Nparticles));
	}
}
