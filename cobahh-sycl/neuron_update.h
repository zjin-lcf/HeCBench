#include "common.h"

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
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1>  d_h  (_ptr_array_neurongroup_h, _N);
  buffer<float, 1>  d_m  (_ptr_array_neurongroup_m, _N);
  buffer<float, 1>  d_n  (_ptr_array_neurongroup_n, _N);
  buffer<float, 1>  d_ge (_ptr_array_neurongroup_ge, _N);
  buffer<float, 1>  d_v  (_ptr_array_neurongroup_v, _N);
  buffer<float, 1>  d_gi (_ptr_array_neurongroup_gi, _N);
  buffer<float, 1>  d_lastspike (_ptr_array_neurongroup_lastspike, _N);
  buffer<char, 1>  d_not_refractory (_ptr_array_neurongroup_not_refractory, _N);

  range<1> global_work_size ((_N+255)/256*256);
  range<1> local_work_size (256);

  for (int i = 0; i < iteration; i++) {

    q.submit([&] (handler &h) {
      auto d_h_acc = d_h.get_access<sycl_read_write>(h);
      auto d_m_acc = d_m.get_access<sycl_read_write>(h);
      auto d_n_acc = d_n.get_access<sycl_read_write>(h);
      auto d_ge_acc = d_ge.get_access<sycl_read_write>(h);
      auto d_gi_acc = d_gi.get_access<sycl_read_write>(h);
      auto d_v_acc = d_v.get_access<sycl_read_write>(h);
      auto d_lastspike_acc = d_lastspike.get_access<sycl_read>(h);
      auto d_not_refractory_acc = d_not_refractory.get_access<sycl_discard_write>(h);
      h.parallel_for<class nstep>(nd_range<1>(global_work_size, local_work_size), [=] (nd_item<1> item) {
        int _idx = item.get_global_id(0);
        if (_idx >= _N) return;
        float h = d_h_acc[_idx];
        float m = d_m_acc[_idx];
        float n = d_n_acc[_idx];
        float ge = d_ge_acc[_idx];
        float v = d_v_acc[_idx];
        const double lastspike = d_lastspike_acc[_idx];
        float gi = d_gi_acc[_idx];
        char not_refractory;
        not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
        const float _BA_h = (_lio_2 * cl::sycl::exp(_lio_3 * v))/(((-4.0f)/(0.001f + (_lio_4 * cl::sycl::exp(_lio_5 * v)))) - (_lio_2 * cl::sycl::exp(_lio_3 * v)));
        const float _h = (- _BA_h) + ((_BA_h + h) * cl::sycl::exp(dt * (((-4.0f)/(0.001f + (_lio_4 * cl::sycl::exp(_lio_5 * v)))) - (_lio_2 * cl::sycl::exp(_lio_3 * v)))));
        const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v))))) - ((0.32f * v)/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))));
        const float _m = (- _BA_m) + ((_BA_m + m) * cl::sycl::exp(dt * (((((_lio_11/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * cl::sycl::exp(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * cl::sycl::exp(_lio_15 * v))))))));
        const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v))))) - ((0.032f * v)/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))) + (_lio_21 * cl::sycl::exp(_lio_22 * v))));
        const float _n = (- _BA_n) + ((_BA_n + n) * cl::sycl::exp(dt * (((_lio_20/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * cl::sycl::exp(_lio_5 * v)))) + (_lio_21 * cl::sycl::exp(_lio_22 * v))))));
        const float _ge = _lio_23 * ge;
        const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
        const float _v = (- _BA_v) + ((_BA_v + v) * cl::sycl::exp(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))));
        const float _gi = _lio_33 * gi;

        d_h_acc[_idx] = _h;
        d_m_acc[_idx] = _m;
        d_n_acc[_idx] = _n;
        d_ge_acc[_idx] = _ge;
        d_v_acc[_idx] = _v;
        d_gi_acc[_idx] = _gi;
        d_not_refractory_acc[_idx] = not_refractory;
      });
    });
  }
  q.wait();
}


