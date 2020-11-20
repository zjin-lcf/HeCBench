__global__ void
kernel_normalize_weights (
		float* weights,
		const float* partial_sums,
		float* CDF,
		float* u,
		int* seed,
		const int Nparticles )
{

	__shared__ float u1;
	__shared__ float sumWeights;
	int local_id = threadIdx.x;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(0 == local_id)
		sumWeights = partial_sums[0];
	__syncthreads();
	if(i < Nparticles) {
		weights[i] = weights[i]/sumWeights;
	}
	__syncthreads();
	if(i == 0) {
		CDF[0] = weights[0];
		for(int x = 1; x < Nparticles; x++){
			CDF[x] = weights[x] + CDF[x-1];
		}

		seed[i] = (A*seed[i] + C) % M;
		float p = fabs(seed[i]/((float)M));
		seed[i] = (A*seed[i] + C) % M;
		float q = fabs(seed[i]/((float)M));
		u[0] = (1/((float)(Nparticles))) * 
			(sqrt(-2*log(p))*cos(2*PI*q));
		// do this to allow all threads in all blocks to use the same u1
	}
	__syncthreads();
	if(0 == local_id)
		u1 = u[0];

	__syncthreads();
	if(i < Nparticles)
	{
		u[i] = u1 + i/((float)(Nparticles));
	}
}
