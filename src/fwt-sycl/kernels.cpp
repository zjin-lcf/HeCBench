/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
#define ELEMENTARY_LOG2SIZE 11

void fwtBatch1Kernel(      float *__restrict d_Output, 
                     const float *__restrict d_Input,
                           float *__restrict s_data,
                           int log2N,
                           sycl::nd_item<1> &item)
{
    int lid = item.get_local_id(0);
    int gid = item.get_group(0);
    int gsz = item.get_local_range(0);

    // Handle to thread block group
    const int    N = 1 << log2N;
    const int base = gid << log2N;

    const float *d_Src = d_Input  + base;
    float *d_Dst = d_Output + base;

    for (int pos = lid; pos < N; pos += gsz)
    {
        s_data[pos] = d_Src[pos];
    }

    //Main radix-4 stages
    const int pos = lid;

    for (int stride = N >> 2; stride > 0; stride >>= 2)
    {
        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        item.barrier(sycl::access::fence_space::local_space);
        float D0 = s_data[i0];
        float D1 = s_data[i1];
        float D2 = s_data[i2];
        float D3 = s_data[i3];

        float T;
        T = D0;
        D0         = D0 + D2;
        D2         = T - D2;
        T = D1;
        D1         = D1 + D3;
        D3         = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
    }

    //Do single radix-2 stage for odd power of two
    if (log2N & 1)
    {
        item.barrier(sycl::access::fence_space::local_space);

        for (int pos = lid; pos < N / 2; pos += gsz)
        {
            int i0 = pos << 1;
            int i1 = i0 + 1;

            float D0 = s_data[i0];
            float D1 = s_data[i1];
            s_data[i0] = D0 + D1;
            s_data[i1] = D0 - D1;
        }
    }

    item.barrier(sycl::access::fence_space::local_space);

    for (int pos = lid; pos < N; pos += gsz)
    {
        d_Dst[pos] = s_data[pos];
    }
}

// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
void fwtBatch2Kernel(
          float *__restrict d_Output,
    const float *__restrict d_Input,
    int stride, sycl::nd_item<2> &item)
{
    const int gidx = item.get_group(1);
    const int lidx = item.get_local_id(1);
    const int gszx = item.get_local_range(1);
    const int grps = item.get_group_range(1);
    const int gidy = item.get_group(0);

    const int pos = gidx * gszx + lidx;
    const int   N = gszx * grps * 4;

    const float *d_Src = d_Input  + gidy * N;
    float *d_Dst = d_Output + gidy * N;

    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    float D0 = d_Src[i0];
    float D1 = d_Src[i1];
    float D2 = d_Src[i2];
    float D3 = d_Src[i3];

    float T;
    T = D0;
    D0        = D0 + D2;
    D2        = T - D2;
    T = D1;
    D1        = D1 + D3;
    D3        = T - D3;
    T = D0;
    d_Dst[i0] = D0 + D1;
    d_Dst[i1] = T - D1;
    T = D2;
    d_Dst[i2] = D2 + D3;
    d_Dst[i3] = T - D3;
}

// Put everything together: batched Fast Walsh Transform CPU front-end
void fwtBatchGPU(sycl::queue &q, float *data, int M, int log2N)
{
    const int THREAD_N = 256;

    int N = 1 << log2N;

    sycl::range<2> gws (M, N / (4 * THREAD_N) * THREAD_N);
    sycl::range<2> lws (1, THREAD_N);

    for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2)
    {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class fwt2>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          fwtBatch2Kernel(data, data, N / 4, item);
        });
      });
    }

    sycl::range<1> gws2 (M * N / 4);
    sycl::range<1> lws2 (N/4);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> lmem (sycl::range<1>(N), cgh);
      cgh.parallel_for<class fwt1>(
        sycl::nd_range<1>(gws2, lws2), [=] (sycl::nd_item<1> item) {
        fwtBatch1Kernel( data,
                         data,
                         lmem.get_pointer(),
                         log2N, item);
      });
    });
}

// Modulate two arrays
void modulateGPU(sycl::queue &q, float *a, const float *b, int N)
{
    sycl::range<1> gws (128*256);
    sycl::range<1> lws (256);
    float rcpN = 1.0f / (float)N;

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class modulate>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int        tid = item.get_global_id(0);
        int numThreads = item.get_group_range(0) * item.get_local_range(0);

        for (int pos = tid; pos < N; pos += numThreads)
            a[pos] *= b[pos] * rcpN;
     });
   });
}
