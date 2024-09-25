#include <stdio.h>
#include <sycl/sycl.hpp>
#include <xpu/Macros.h>
#include <xpu/Stream.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/StreamGuard.h>


// require T <= Tmax, T % 4 == 0, B % BF == 0, B % BB === 0 (Tmax and BF and BB are passed by compiler)

#define F4(A, B) ((sycl::float4 *)(A))[(B) >> 2]

inline sycl::queue& getQueue() 
{
  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream xpu_stream = impl.getStream(impl.getDevice());
  return xpu::get_queue_from_stream(xpu_stream);
}

/* template <typename F>
__global__ void kernel_forward(const F *__restrict__ const __w, const F
*__restrict__ const __k, F *__restrict__ const x, const F eps, const int B,
const int C, const int T) {*/
template <typename F>
void kernel_forward(const F *__restrict__ const __w,
                    const F *__restrict__ const __k, F *__restrict__ const x,
                    const F eps, const int B, const int C, const int T,
                    const sycl::nd_item<3> &item, F *ww, F *kk) {
/* const int i = blockIdx.y;*/
    const int i = item.get_group(1);
/* const int t = threadIdx.x << 2;*/
    const int t = item.get_local_id(2) << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BF;

    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BF; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
    }
    item.barrier(sycl::access::fence_space::local_space);

    sycl::float4 ss[BF];
#pragma unroll
    for (int j = 0; j < BF; j++) {
        ss[j] = {eps, eps, eps, eps};
    }
    for (int u = 0; u <= t; u++) {
        const F *__restrict__ const w = ww + T - t + u - 4;
        #pragma unroll
        for (int j = 0; j < BF; j++) {
            sycl::float4 *__restrict__ const s = ss + j;
            const F k = kk[u + Tmax * j];
            s->x() += w[3] * k;
            s->y() += w[2] * k;
            s->z() += w[1] * k;
            s->w() += w[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BF; j++) {
        sycl::float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const w = ww + T - 3;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        s->y() += w[2] * k[0];
        s->z() += w[1] * k[0];
        s->z() += w[2] * k[1];
        s->w() += w[0] * k[0];
        s->w() += w[1] * k[1];
        s->w() += w[2] * k[2];
        F4(x, ti + tj * j) = *s;
    }
}

template <typename F>
void kernel_backward(const F *__restrict__ const __w,
                     const F *__restrict__ const __k,
                     const F *__restrict__ const __gwk,
                     F *__restrict__ const gw, F *__restrict__ const gk,
                     const int B, const int C, const int T,
                     const sycl::nd_item<3> &item, F *ww, F *kk, F *gg) {
    const int i = item.get_group(1);
    const int t = item.get_local_id(2) << 2;
    const int ti = t + T * i;
    const int tj = T * (B * C) / BB;

    F4(ww, t) = F4(__w, t + T * (i % C));

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        F4(kk, t + Tmax * j) = F4(__k, ti + tj * j);
        F4(gg, t + Tmax * j) = F4(__gwk, ti + tj * j);
    }
    item.barrier(sycl::access::fence_space::local_space);

    sycl::float4 ss[BB];
#pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = 0; u <= t; u++) {
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            sycl::float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T - t + u - 4;
            const F k = kk[u + Tmax * j];
            s->x() += g[3] * k;
            s->y() += g[2] * k;
            s->z() += g[1] * k;
            s->w() += g[0] * k;
        }
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        sycl::float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const k = kk + Tmax * j + t + 1;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        s->y() += g[2] * k[0];
        s->z() += g[1] * k[0];
        s->z() += g[2] * k[1];
        s->w() += g[0] * k[0];
        s->w() += g[1] * k[1];
        s->w() += g[2] * k[2];
        F4(gw, ti + tj * j) = *s;
    }

    #pragma unroll
    for (int j = 0; j < BB; j++) {
        ss[j] = {0, 0, 0, 0};
    }
    for (int u = t + 3; u < T; u++) {
        const F w = ww[u];
        #pragma unroll
        for (int j = 0; j < BB; j++) {
            sycl::float4 *__restrict__ const s = ss + j;
            const F *__restrict__ const g = gg + Tmax * j + T + t - u - 1;
            s->x() += g[0] * w;
            s->y() += g[1] * w;
            s->z() += g[2] * w;
            s->w() += g[3] * w;
        }        
    }
    #pragma unroll
    for (int j = 0; j < BB; j++) {
        sycl::float4 *__restrict__ const s = ss + j;
        const F *__restrict__ const g = gg + Tmax * j + T - 3;
        const F *__restrict__ const w = ww + t;
        s->x() += g[2] * w[0];
        s->x() += g[1] * w[1];
        s->x() += g[0] * w[2];
        s->y() += g[2] * w[1];
        s->y() += g[1] * w[2];
        s->z() += g[2] * w[2];
        F4(gk, ti + tj * j) = *s;
    }
}

void gpu_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T) {
    sycl::range<3> gridDim(1, B * C / BF, 1);
    sycl::range<3> blockDim(1, 1, T >> 2);
    getQueue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> ww_acc(sycl::range<1>(Tmax), cgh);
        sycl::local_accessor<float, 1> kk_acc(sycl::range<1>(Tmax * BF), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(gridDim * blockDim, blockDim),
            [=](sycl::nd_item<3> item) {
                kernel_forward(
                    w, k, x, eps, B, C, T, item,
                    ww_acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
                    kk_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}

void gpu_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T) {
    sycl::range<3> gridDim(1, B * C / BB, 1);
    sycl::range<3> blockDim(1, 1, T >> 2);
    getQueue().submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> ww_acc(sycl::range<1>(Tmax), cgh);
        sycl::local_accessor<float, 1> kk_acc(sycl::range<1>(Tmax * BB), cgh);
        sycl::local_accessor<float, 1> gg_acc(sycl::range<1>(Tmax * BB), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(gridDim * blockDim, blockDim),
            [=](sycl::nd_item<3> item) {
                kernel_backward(
                    w, k, gwk, gw, gk, B, C, T, item,
                    ww_acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
                    kk_acc.template get_multi_ptr<sycl::access::decorated::no>().get(),
                    gg_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
            });
    });
}
