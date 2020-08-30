#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdio.h>

#define N 512
#define ITERATION 100000


template<typename dataType>
void prescan(dataType *g_odata, dataType *g_idata, int n,
             sycl::nd_item<3> item_ct1, dataType *temp)
{

 int thid = item_ct1.get_local_id(2);
        int offset = 1;
	temp[2*thid]   = g_idata[2*thid];
	temp[2*thid+1] = g_idata[2*thid+1];
	for (int d = n >> 1; d > 0; d >>= 1) 
	{
  item_ct1.barrier();
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
  item_ct1.barrier();
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
void runTest (dataType *in, dataType *out, int n)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        dataType *d_in;
	dataType *d_out;
 d_in = (dataType *)sycl::malloc_device(N * sizeof(dataType), q_ct1);
 d_out = (dataType *)sycl::malloc_device(N * sizeof(dataType), q_ct1);
 q_ct1.memcpy(d_in, in, N * sizeof(dataType)).wait();
        for (int i = 0; i < ITERATION; i++) {
  q_ct1.submit([&](sycl::handler &cgh) {
   sycl::accessor<dataType, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       temp_acc_ct1(sycl::range<1>(512 /*N*/), cgh);

   cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, N / 2),
                                      sycl::range<3>(1, 1, N / 2)),
                    [=](sycl::nd_item<3> item_ct1) {
                     prescan(d_out, d_in, n, item_ct1,
                             (dataType *)temp_acc_ct1.get_pointer());
                    });
  });
        }
 q_ct1.memcpy(out, d_out, N * sizeof(dataType)).wait();
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



