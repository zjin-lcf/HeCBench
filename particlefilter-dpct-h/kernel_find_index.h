#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void kernel_find_index(const float *arrayX, const float *arrayY,
                       const float *CDF, const float *u, float *xj, float *yj,
                       const int Nparticles, sycl::nd_item<3> item_ct1)
{
 int i = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
         item_ct1.get_local_id(2);
        if(i < Nparticles){
		int index = -1;
		int x;

		for(x = 0; x < Nparticles; x++){
			if(CDF[x] >= u[i]){
				index = x;
				break;
			}
		}
		if(index == -1){
			index = Nparticles-1;
		}

		xj[i] = arrayX[index];
		yj[i] = arrayY[index];


	}
}
