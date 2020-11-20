__global__ void
kernel_find_index (
		const float* arrayX,
		const float* arrayY,
		const float* CDF,
		const float* u,
		float* xj,
		float* yj,
		const int Nparticles)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
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
