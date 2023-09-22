#include <chrono>
#include <sycl/sycl.hpp>

inline int _timestep(float t, float dt)
{
  return (int)((t + 1e-3f*dt)/dt); 
}

void neurongroup_stateupdater (
    float* __restrict  _ptr_array_neurongroup_ge,
    float* __restrict  _ptr_array_neurongroup_gi,
    float* __restrict  _ptr_array_neurongroup_h,
    float* __restrict  _ptr_array_neurongroup_m,
    float* __restrict  _ptr_array_neurongroup_n,
    float* __restrict  _ptr_array_neurongroup_v,
    float* __restrict   _ptr_array_neurongroup_lastspike,
    float* __restrict  _ptr_array_defaultclock_dt,
    float*__restrict  _ptr_array_defaultclock_t,
    char* __restrict  _ptr_array_neurongroup_not_refractory,
    const int _N,
    const int iteration ) 
{
  const float dt = _ptr_array_defaultclock_dt[0];
  const float t = _ptr_array_defaultclock_t[0];
  const int    _lio_1 = _timestep(0.003, dt);
  const float  _lio_2 = 9.939082f; //1.0f*(0.3291372 * exp(1.0f*(0.055555556 * (-0.063f))/0.001f))/0.001f;
  const float  _lio_3 = -55.555556f; //1.0f*(-0.055555556)/0.001f;
  const float  _lio_4 = 0.00001f; //2980.958 * (0.001f * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float  _lio_5 = -200.0f; //1.0f*(-0.2)/0.001f;
  const float  _lio_6 = -0.02016f; //0.32 * (-0.063f);
  const float  _lio_7 = -0.000001f; //- 0.001f * 0.001f;
  const float  _lio_8 = 0; //25.79034 * ((0.001f * 0.001f) * exp(1.0f*(0.25 * (-0.063f))/0.001f));
  const float  _lio_9 = -250.0f; //1.0f*(-0.25)/0.001f;
  const float _lio_10 = 0.00416f; //4.16 * 0.001f;
  const float _lio_11 = 0.02016f; //(-0.32) * (-0.063f);
  const float _lio_12 = -0.01764f; //0.28 * (-0.063f);
  const float _lio_13 = -0.000001f; //(-1.0) * (0.001f * 0.001f);
  const float _lio_14 = 0.000099f; //0.00033546262 * ((0.001f * 0.001f) * exp(1.0f*((-0.2) * (-0.063f))/0.001f));
  const float _lio_15 = 200.0f; //1.0f*0.2/0.001f;
  const float _lio_16 = 0.0112f; //11.2 * 0.001f;
  const float _lio_17 = -0.002016f; //0.032 * (-0.063f);
  const float _lio_18 = 0; //20.085537 * ((0.001f * 0.001f) * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float _lio_19 = 0.00048f; //0.48 * 0.001f;
  const float _lio_20 = 0.002016f; //(-0.032) * (-0.063f);
  const float _lio_21 = 132.901474f; //1.0f*(0.6420127 * exp(1.0f*(0.025 * (-0.063f))/0.001f))/0.001f;
  const float _lio_22 = -25.0f ;//1.0f*(-0.025)/0.001f;
  const float _lio_23 = expf(-2000.0f*dt);
  const float _lio_24 = -3.0f; //1.0f*((-0.06f) * 1e-08f)/2e-10f;
  const float _lio_25 = -2700.0f; //1.0f*((-0.09f) * 6e-06f)/2e-10f;
  const float _lio_26 = 5000.0f; //1.0f*(0.05f * 2e-05f)/2e-10f;
  const float _lio_27 = 0; // 1.0f*0.0f/2e-10f;
  const float _lio_28 = -400000000.0f; //1.0f*(-0.08f)/2e-10f;
  const float _lio_29 = -50.0f; //0.0 - (1.0f*1e-08f/2e-10f);
  const float _lio_30 = -30000.0f; //1.0f*(- 6e-06f)/2e-10f;
  const float _lio_31 = 100000.0f; //1.0f*2e-05f/2e-10f;
  const float _lio_32 = 5000000000.0f; //1.0f*1.0/2e-10f;
  const float _lio_33 = expf(-100.0f*dt);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t size = _N * sizeof(float);

  float *d_h  = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_h, _ptr_array_neurongroup_h, size);

  float *d_m  = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_m, _ptr_array_neurongroup_m, size);

  float *d_n  = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_n, _ptr_array_neurongroup_n, size);

  float *d_ge = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_ge, _ptr_array_neurongroup_ge, size);

  float *d_v  = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_v, _ptr_array_neurongroup_v, size); 

  float *d_gi = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_gi, _ptr_array_neurongroup_gi, size); 

  float *d_lastspike = sycl::malloc_device<float>(_N, q);
  q.memcpy(d_lastspike, _ptr_array_neurongroup_lastspike, size); 

  char *d_not_refractory = sycl::malloc_device<char>(_N, q);

  sycl::range<1> gws ((_N+255)/256*256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iteration; i++) {
    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class nstep>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int _idx = item.get_global_id(0);
        if (_idx >= _N) return;
        float h = d_h[_idx];
        float m = d_m[_idx];
        float n = d_n[_idx];
        float ge = d_ge[_idx];
        float v = d_v[_idx];
        const float lastspike = d_lastspike[_idx];
        float gi = d_gi[_idx];
        char not_refractory;
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const float _BA_h = (_lio_2 * sycl::exp(_lio_3 * v))/(((-4.0f)/(0.001f + (_lio_4 * sycl::exp(_lio_5 * v)))) - (_lio_2 * sycl::exp(_lio_3 * v)));
        const float _h = (- _BA_h) + ((_BA_h + h) * sycl::exp(dt * (((-4.0f)/(0.001f + (_lio_4 * sycl::exp(_lio_5 * v)))) - (_lio_2 * sycl::exp(_lio_3 * v)))));
        const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) - ((0.32f * v)/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))));
        const float _m = (- _BA_m) + ((_BA_m + m) * sycl::exp(dt * (((((_lio_11/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))))));
        const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) - ((0.032f * v)/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) + (_lio_21 * sycl::exp(_lio_22 * v))));
        const float _n = (- _BA_n) + ((_BA_n + n) * sycl::exp(dt * (((_lio_20/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) + (_lio_21 * sycl::exp(_lio_22 * v))))));
        const float _ge = _lio_23 * ge;
        const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
        const float _v = (- _BA_v) + ((_BA_v + v) * sycl::exp(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))));
        const float _gi = _lio_33 * gi;

        d_h[_idx] = _h;
        d_m[_idx] = _m;
        d_n[_idx] = _n;
        d_ge[_idx] = _ge;
        d_v[_idx] = _v;
        d_gi[_idx] = _gi;
        d_not_refractory[_idx] = not_refractory;
      });
    });
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iteration);

  q.memcpy(_ptr_array_neurongroup_ge, d_ge, size);
  q.memcpy(_ptr_array_neurongroup_gi, d_gi, size);
  q.memcpy(_ptr_array_neurongroup_m, d_m, size);
  q.memcpy(_ptr_array_neurongroup_n, d_n, size);
  q.memcpy(_ptr_array_neurongroup_v, d_v, size);
  q.memcpy(_ptr_array_neurongroup_h, d_h, size);
  q.memcpy(_ptr_array_neurongroup_not_refractory, d_not_refractory, _N*sizeof(char));
  q.wait();

  sycl::free(d_h, q);
  sycl::free(d_m, q);
  sycl::free(d_n, q);
  sycl::free(d_ge, q);
  sycl::free(d_gi, q);
  sycl::free(d_v, q);
  sycl::free(d_lastspike, q);
  sycl::free(d_not_refractory, q);
}
