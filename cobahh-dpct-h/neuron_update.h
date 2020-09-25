#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>

inline int _timestep(float t, float dt)
{
  return (int)((t + 1e-3f*dt)/dt); 
}

void cobahh (
    float* d_h, float* d_m, float* d_n, float* d_ge,
    float* d_v, float* d_gi, const float* d_lastspike, 
    char* d_not_refractory, const int _N ,
    const float dt,
    const float t,
    const int    _lio_1,
    const float  _lio_2,
    const float  _lio_3,
    const float  _lio_4,
    const float  _lio_5,
    const float  _lio_6,
    const float  _lio_7,
    const float  _lio_8,
    const float  _lio_9,
    const float _lio_10,
    const float _lio_11,
    const float _lio_12,
    const float _lio_13,
    const float _lio_14,
    const float _lio_15,
    const float _lio_16,
    const float _lio_17,
    const float _lio_18,
    const float _lio_19,
    const float _lio_20,
    const float _lio_21,
    const float _lio_22,
    const float _lio_23,
    const float _lio_24,
    const float _lio_25,
    const float _lio_26,
    const float _lio_27,
    const float _lio_28,
    const float _lio_29,
    const float _lio_30,
    const float _lio_31,
    const float _lio_32,
    const float _lio_33
    , sycl::nd_item<3> item_ct1)

{
  int _idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2);
  if (_idx >= _N) return;
  float h = d_h[_idx];
  float m = d_m[_idx];
  float n = d_n[_idx];
  float ge = d_ge[_idx];
  float v = d_v[_idx];
  const double lastspike = d_lastspike[_idx];
  float gi = d_gi[_idx];
  char not_refractory;
  not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
  const float _BA_h = (_lio_2 * sycl::exp(_lio_3 * v)) /
                      (((-4.0f) / (0.001f + (_lio_4 * sycl::exp(_lio_5 * v)))) -
                       (_lio_2 * sycl::exp(_lio_3 * v)));
  const float _h =
      (-_BA_h) +
      ((_BA_h + h) *
       sycl::exp(dt * (((-4.0f) / (0.001f + (_lio_4 * sycl::exp(_lio_5 * v)))) -
                       (_lio_2 * sycl::exp(_lio_3 * v)))));
  const float _BA_m =
      (((_lio_6 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) +
        (_lio_10 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) -
       ((0.32f * v) / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) /
      (((((_lio_11 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) +
          (_lio_12 / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) +
         (_lio_16 / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) +
        ((0.32f * v) / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) -
       ((_lio_10 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) +
        ((0.28f * v) / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))));
  const float _m =
      (-_BA_m) +
      ((_BA_m + m) *
       sycl::exp(
           dt *
           (((((_lio_11 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) +
               (_lio_12 / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) +
              (_lio_16 / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))) +
             ((0.32f * v) / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v))))) -
            ((_lio_10 / (_lio_7 + (_lio_8 * sycl::exp(_lio_9 * v)))) +
             ((0.28f * v) / (_lio_13 + (_lio_14 * sycl::exp(_lio_15 * v))))))));
  const float _BA_n =
      (((_lio_17 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) +
        (_lio_19 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) -
       ((0.032f * v) / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) /
      (((_lio_20 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) +
        ((0.032f * v) / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) -
       ((_lio_19 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) +
        (_lio_21 * sycl::exp(_lio_22 * v))));
  const float _n =
      (-_BA_n) +
      ((_BA_n + n) *
       sycl::exp(
           dt *
           (((_lio_20 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) +
             ((0.032f * v) / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v))))) -
            ((_lio_19 / (_lio_7 + (_lio_18 * sycl::exp(_lio_5 * v)))) +
             (_lio_21 * sycl::exp(_lio_22 * v))))));
  const float _ge = _lio_23 * ge;
  const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
  const float _v =
      (-_BA_v) +
      ((_BA_v + v) *
       sycl::exp(dt * ((_lio_29 + (_lio_30 * (n * n * n * n))) -
                       (((_lio_31 * (h * (m * m * m))) + (_lio_32 * ge)) +
                        (_lio_32 * gi)))));
  const float _gi = _lio_33 * gi;

  d_h[_idx] = _h;
  d_m[_idx] = _m;
  d_n[_idx] = _n;
  d_ge[_idx] = _ge;
  d_v[_idx] = _v;
  d_gi[_idx] = _gi;
  d_not_refractory[_idx] = not_refractory;
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
  const float _lio_23 = expf(-2000.0f * dt);
  const float _lio_24 = -3.0f; //1.0f*((-0.06f) * 1e-08f)/2e-10f;
  const float _lio_25 = -2700.0f; //1.0f*((-0.09f) * 6e-06f)/2e-10f;
  const float _lio_26 = 5000.0f; //1.0f*(0.05f * 2e-05f)/2e-10f;
  const float _lio_27 = 0; // 1.0f*0.0f/2e-10f;
  const float _lio_28 = -400000000.0f; //1.0f*(-0.08f)/2e-10f;
  const float _lio_29 = -50.0f; //0.0 - (1.0f*1e-08f/2e-10f);
  const float _lio_30 = -30000.0f; //1.0f*(- 6e-06f)/2e-10f;
  const float _lio_31 = 100000.0f; //1.0f*2e-05f/2e-10f;
  const float _lio_32 = 5000000000.0f; //1.0f*1.0/2e-10f;
  const float _lio_33 = expf(-100.0f * dt);

  float* d_h;
  dpct::dpct_malloc((void **)&d_h, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_h, _ptr_array_neurongroup_h, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_m;
  dpct::dpct_malloc((void **)&d_m, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_m, _ptr_array_neurongroup_m, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_n;
  dpct::dpct_malloc((void **)&d_n, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_n, _ptr_array_neurongroup_n, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_ge;
  dpct::dpct_malloc((void **)&d_ge, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_ge, _ptr_array_neurongroup_ge, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_v;
  dpct::dpct_malloc((void **)&d_v, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_v, _ptr_array_neurongroup_v, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_gi;
  dpct::dpct_malloc((void **)&d_gi, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_gi, _ptr_array_neurongroup_gi, _N * sizeof(float),
                          dpct::host_to_device);

  float* d_lastspike;
  dpct::dpct_malloc((void **)&d_lastspike, _N * sizeof(float));
  dpct::async_dpct_memcpy(d_lastspike, _ptr_array_neurongroup_lastspike,
                          _N * sizeof(float), dpct::host_to_device);

  char* d_not_refractory;
  dpct::dpct_malloc((void **)&d_not_refractory, _N * sizeof(char));

  sycl::range<3> grids((_N + 255) / 256, 1, 1);
  sycl::range<3> threads(256, 1, 1);

  for (int n = 0; n < iteration; n++) {

    dpct::buffer_t d_h_buf_ct0 = dpct::get_buffer(d_h);
    dpct::buffer_t d_m_buf_ct1 = dpct::get_buffer(d_m);
    dpct::buffer_t d_n_buf_ct2 = dpct::get_buffer(d_n);
    dpct::buffer_t d_ge_buf_ct3 = dpct::get_buffer(d_ge);
    dpct::buffer_t d_v_buf_ct4 = dpct::get_buffer(d_v);
    dpct::buffer_t d_gi_buf_ct5 = dpct::get_buffer(d_gi);
    dpct::buffer_t d_lastspike_buf_ct6 = dpct::get_buffer(d_lastspike);
    dpct::buffer_t d_not_refractory_buf_ct7 =
        dpct::get_buffer(d_not_refractory);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto d_h_acc_ct0 =
          d_h_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_m_acc_ct1 =
          d_m_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_n_acc_ct2 =
          d_n_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
      auto d_ge_acc_ct3 =
          d_ge_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto d_v_acc_ct4 =
          d_v_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
      auto d_gi_acc_ct5 =
          d_gi_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
      auto d_lastspike_acc_ct6 =
          d_lastspike_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
      auto d_not_refractory_acc_ct7 =
          d_not_refractory_buf_ct7.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            cobahh((float *)(&d_h_acc_ct0[0]), (float *)(&d_m_acc_ct1[0]),
                   (float *)(&d_n_acc_ct2[0]), (float *)(&d_ge_acc_ct3[0]),
                   (float *)(&d_v_acc_ct4[0]), (float *)(&d_gi_acc_ct5[0]),
                   (const float *)(&d_lastspike_acc_ct6[0]),
                   (char *)(&d_not_refractory_acc_ct7[0]), _N, dt, t, _lio_1,
                   _lio_2, _lio_3, _lio_4, _lio_5, _lio_6, _lio_7, _lio_8,
                   _lio_9, _lio_10, _lio_11, _lio_12, _lio_13, _lio_14, _lio_15,
                   _lio_16, _lio_17, _lio_18, _lio_19, _lio_20, _lio_21,
                   _lio_22, _lio_23, _lio_24, _lio_25, _lio_26, _lio_27,
                   _lio_28, _lio_29, _lio_30, _lio_31, _lio_32, _lio_33,
                   item_ct1);
          });
    });
  }

  dpct::async_dpct_memcpy(_ptr_array_neurongroup_ge, d_ge, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_gi, d_gi, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_m, d_m, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_n, d_n, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_v, d_v, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_h, d_h, _N * sizeof(float),
                          dpct::device_to_host);
  dpct::async_dpct_memcpy(_ptr_array_neurongroup_not_refractory,
                          d_not_refractory, _N * sizeof(char),
                          dpct::device_to_host);
  dpct::get_current_device().queues_wait_and_throw();

  dpct::dpct_free(d_h);
  dpct::dpct_free(d_m);
  dpct::dpct_free(d_n);
  dpct::dpct_free(d_ge);
  dpct::dpct_free(d_gi);
  dpct::dpct_free(d_v);
  dpct::dpct_free(d_lastspike);
  dpct::dpct_free(d_not_refractory);
}


