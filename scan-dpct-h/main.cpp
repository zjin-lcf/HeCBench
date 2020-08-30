#define DPCT_USM_LEVEL_NONE
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
	dataType *d_in;
	dataType *d_out;
 dpct::dpct_malloc((void **)&d_in, N * sizeof(dataType));
 dpct::dpct_malloc((void **)&d_out, N * sizeof(dataType));
 dpct::dpct_memcpy(d_in, in, N * sizeof(dataType), dpct::host_to_device);
        for (int i = 0; i < ITERATION; i++) {
  dpct::buffer_t d_out_buf_ct0 = dpct::get_buffer(d_out);
  std::pair<dpct::buffer_t, size_t> d_in_buf_ct1 =
      dpct::get_buffer_and_offset(d_in);
  size_t d_in_offset_ct1 = d_in_buf_ct1.second;
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
   sycl::accessor<dataType, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       temp_acc_ct1(sycl::range<1>(512 /*N*/), cgh);
   auto d_out_acc_ct0 =
       d_out_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
   auto d_in_acc_ct1 =
       d_in_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);

   cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, N / 2),
                                      sycl::range<3>(1, 1, N / 2)),
                    [=](sycl::nd_item<3> item_ct1) {
                     dataType *d_in_ct1 =
                         (dataType *)(&d_in_acc_ct1[0] + d_in_offset_ct1);
                     prescan((dataType *)(&d_out_acc_ct0[0]), d_in_ct1, n,
                             item_ct1, (dataType *)temp_acc_ct1.get_pointer());
                    });
  });
        }
 dpct::dpct_memcpy(out, d_out, N * sizeof(dataType), dpct::device_to_host);
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



