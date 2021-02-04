#include <hip/hip_runtime.h>
#include <assert.h>
#include <stdio.h>

#define N 512
#define ITERATION 100000


template<typename dataType>
__global__ void prescan(dataType *g_odata, const dataType *g_idata, const int n)
{
	__shared__ dataType temp[N];
	int thid = threadIdx.x; 
	int offset = 1;
	temp[2*thid]   = g_idata[2*thid];
	temp[2*thid+1] = g_idata[2*thid+1];
	for (int d = n >> 1; d > 0; d >>= 1) 
	{
		__syncthreads();
		if (thid < d) 
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) temp[n-1] = 0; // clear the last elem
	for (int d = 1; d < n; d *= 2) // traverse down
	{
		offset >>= 1;     
	       	__syncthreads();      
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	g_odata[2*thid] = temp[2*thid];
	g_odata[2*thid+1] = temp[2*thid+1];
}

template <typename dataType>
void runTest (const dataType *in, dataType *out, const int n) 
{
	dataType *d_in;
	dataType *d_out;
	hipMalloc((void**)&d_in, N*sizeof(dataType));
	hipMalloc((void**)&d_out, N*sizeof(dataType));
	hipMemcpy(d_in, in, N*sizeof(dataType), hipMemcpyHostToDevice); 
        for (int i = 0; i < ITERATION; i++) {
	  hipLaunchKernelGGL(prescan, dim3(1), dim3(N/2), 0, 0, d_out, d_in, n);
	}
	hipMemcpy(out, d_out, N*sizeof(dataType), hipMemcpyDeviceToHost);
}

int main() 
{
	float in[N];
	float cpu_out[N];
	float gpu_out[N];
	int error = 0;

	for (int i = 0; i < N; i++) in[i] = (i % 5)+1;

	runTest(in, gpu_out, N);

	cpu_out[0] = 0;
	if (gpu_out[0] != 0) {
		error++;
		printf("gpu = %f at index 0\n", gpu_out[0]);
	}
	for (int i = 1; i < N; i++) 
	{
		cpu_out[i] = cpu_out[i-1] + in[i-1];
		if (cpu_out[i] != gpu_out[i]) {
			error++;
			printf("cpu = %f gpu = %f at index %d\n",
					cpu_out[i], gpu_out[i], i);
		}
	}

	if (error == 0) printf("PASS\n");
	return 0;

}



