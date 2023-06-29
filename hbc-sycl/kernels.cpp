#include <vector>
#include <iostream>
#include "util.h" // graph

#define DIAMETER_SAMPLES 512

inline int atomicCAS(int &val, int expected, int desired)
{
  int expected_value = expected;
  auto atm = sycl::atomic_ref<int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline T atomicAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed,
     MemoryScope, sycl::access::address_space::global_space> ref(val);
  return ref.fetch_add(delta);
}

template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::work_group>
static inline T atomicAddLocal(T& val, const T delta)
{
  sycl::atomic_ref<T, sycl::memory_order::relaxed,
     MemoryScope, sycl::access::address_space::local_space> ref(val);
  return ref.fetch_add(delta);
}

//Note: N must be a power of two
//Simple/Naive bitonic sort. We're only sorting ~512 elements one time, so performance isn't important
void bitonic_sort(int *values, const int N, sycl::nd_item<1> item)
{
  unsigned int idx = item.get_local_id(0);

  for (int k = 2; k <= N; k <<= 1)
  {
    for (int j = k >> 1; j > 0; j = j >> 1)
    {
      while(idx < N)
      {
        int ixj = idx^j;
        if (ixj > idx)
        {
          if ((idx&k) == 0 && values[idx] > values[ixj])
          {
            //exchange(idx, ixj);
            int tmp = values[idx];
            values[idx] = values[ixj];
            values[ixj] = tmp;
          }
          if ((idx&k) != 0 && values[idx] < values[ixj])
          {
            //exchange(idx, ixj);
            int tmp = values[idx];
            values[idx] = values[ixj];
            values[ixj] = tmp;
          }
        }
        idx += item.get_local_range(0);
      }
      item.barrier(sycl::access::fence_space::local_space);
      idx = item.get_local_id(0);
    }
  }
}

void bc_kernel(
  float *__restrict bc,
  const int *__restrict R,
  const int *__restrict C,
  const int *__restrict F,
  const int n,
  const int m,
  const int *__restrict d,
  const unsigned long long *__restrict sigma,
  const float *__restrict delta,
  const int *__restrict Q,
  const int *__restrict Q2,
  const int *__restrict S,
  const int *__restrict endpoints,
  int *__restrict next_source,
  const size_t pitch_d,
  const size_t pitch_sigma,
  const size_t pitch_delta,
  const size_t pitch_Q,
  const size_t pitch_Q2,
  const size_t pitch_S,
  const size_t pitch_endpoints,
  const int start,
  const int end,
  int *__restrict jia,
  int *__restrict diameters,
  const int *__restrict source_vertices,
  const bool approx,
  sycl::nd_item<1> &item,
  int &ind,
  int &i,
  int *&Q_row,
  int *&Q2_row,
  int *&S_row,
  int *&endpoints_row,
  int &Q_len,
  int &Q2_len,
  int &S_len,
  int &current_depth,
  int &endpoints_len,
  bool &sp_calc_done,
  int &next_index,
  int *diameter_keys)
{

  int j = item.get_local_id(0);
  int *d_row = (int *)((char *)d + item.get_group(0) * pitch_d);
  unsigned long long *sigma_row = (unsigned long long *)((char *)sigma + item.get_group(0) * pitch_sigma);
  float *delta_row = (float *)((char *)delta + item.get_group(0) * pitch_delta);
  if(j == 0)
  {
    ind = item.get_group(0) + start;
    i = approx ? source_vertices[(ind)] : ind;
    Q_row = (int *)((char *)Q + item.get_group(0) * pitch_Q);
    Q2_row = (int *)((char *)Q2 + item.get_group(0) * pitch_Q2);
    S_row = (int *)((char *)S + item.get_group(0) * pitch_S);
    endpoints_row = (int *)((char *)endpoints + item.get_group(0) * pitch_endpoints);
    *jia = 0;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if ((ind == 0) && (j < DIAMETER_SAMPLES))
  {
    diameters[j] = INT_MAX;
  }
  item.barrier(sycl::access::fence_space::local_space);

  while (ind < end)
  {
    //Initialization
    for (int k = item.get_local_id(0); k < n; k += item.get_local_range(0))
    {
      if (k == i) // If k is the source node...
      {
        d_row[k] = 0;
        sigma_row[k] = 1;
      }
      else
      {
        d_row[k] = INT_MAX;
        sigma_row[k] = 0;
      }
      delta_row[k] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    //Shortest Path Calculation

    if(j == 0)
    {
      Q_row[0] = i;
      Q_len = 1;
      Q2_len = 0;
      S_row[0] = i;
      S_len = 1;
      endpoints_row[0] = 0;
      endpoints_row[1] = 1;
      endpoints_len = 2;
      current_depth = 0;
      sp_calc_done = false;
    }
    item.barrier(sycl::access::fence_space::local_space);

    //Do first iteration separately since we already know the edges to traverse
    for (int r = item.get_local_id(0) + R[i]; r < R[i + 1]; r += item.get_local_range(0))
    {
      int w = C[r];
      //No multiple/self edges - each value of w is unique, so no need for atomics
      if(d_row[w] == INT_MAX)
      {
        d_row[w] = 1;
        int t = atomicAddLocal(Q2_len, 1);
        Q2_row[t] = w;
      }
      if (d_row[w] == (d_row[i] + 1))
      {
        atomicAdd(sigma_row[w], 1ULL);
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (Q2_len == 0)
    {
      sp_calc_done = true;
    }
    else
    {
      for (int kk = item.get_local_id(0); kk < Q2_len; kk += item.get_local_range(0))
      {
        Q_row[kk] = Q2_row[kk];
        S_row[kk + S_len] = Q2_row[kk];
      }
      item.barrier(sycl::access::fence_space::local_space);
      if(j == 0)
      {
        endpoints_row[endpoints_len] = endpoints_row[endpoints_len - 1] + Q2_len;
        endpoints_len++;
        Q_len = Q2_len;
        S_len += Q2_len;
        Q2_len = 0;
        current_depth++;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    while (!sp_calc_done)
    {
      if ((*jia) && (Q_len > 512))
      {
        for (int k = item.get_local_id(0); k < 2 * m; k += item.get_local_range(0))
        {
          int v = F[k];
          if (d_row[v] == current_depth)
          {
            int w = C[k];
            if(atomicCAS(d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
            {
              int t = atomicAddLocal(Q2_len, 1);
              Q2_row[t] = w;
            }
            if(d_row[w] == (d_row[v]+1))
            {
              atomicAdd(sigma_row[w], sigma_row[v]);
            }
          }
        }
      }
      else
      {

        if(j == 0)
        {
          next_index = item.get_local_range(0);
        }
        item.barrier(sycl::access::fence_space::local_space);
        int k = item.get_local_id(0); // Initial vertices
        while (k < Q_len)
        {
          int v = Q_row[k];
          for(int r=R[v]; r<R[v+1]; r++)
          {
            int w = C[r];
            //Use atomicCAS to prevent duplicates
            if(atomicCAS(d_row[w],INT_MAX,d_row[v]+1) == INT_MAX)
            {
              int t = atomicAddLocal(Q2_len, 1);
              Q2_row[t] = w;
            }
            if(d_row[w] == (d_row[v]+1))
            {
              atomicAdd(sigma_row[w], sigma_row[v]);
            }
          }
          k = atomicAddLocal(next_index, 1);
        }
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (Q2_len == 0) // If there is no additional work found, we're
                              // done
      {
        break;
      }
      else //If there is additional work, transfer elements from Q2 to Q, reset lengths, and add vertices to the stack
      {
        for (int kk = item.get_local_id(0); kk < Q2_len; kk += item.get_local_range(0))
        {
          Q_row[kk] = Q2_row[kk];
          S_row[kk + S_len] = Q2_row[kk];
        }
        item.barrier(sycl::access::fence_space::local_space);
        if(j == 0)
        {
          endpoints_row[endpoints_len] = endpoints_row[endpoints_len - 1] + Q2_len;
          endpoints_len++;
          Q_len = Q2_len;
          S_len += Q2_len;
          Q2_len = 0;
          current_depth++;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
    }

    //The elements at the end of the stack will have the largest distance from the source
    //Using the successor method, we can start from one depth earlier
    if(j == 0)
    {
      current_depth = d_row[S_row[S_len - 1]] - 1;
      if (ind < DIAMETER_SAMPLES)
      {
        diameters[(ind)] = current_depth + 1;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    //Dependency Accumulation (Madduri/Ediger successor method)
    while (current_depth > 0)
    {
      int stack_iter_len = endpoints_row[current_depth + 1] -
                           endpoints_row[(current_depth)];
      if((*jia) && (stack_iter_len>512))
      {
        for (int kk = item.get_local_id(0); kk < 2 * m; kk += item.get_local_range(0)) {
          int w = F[kk];
          if (d_row[w] == current_depth)
          {
            int v = C[kk];
            if(d_row[v] == (d_row[w]+1))
            {
              float change = (sigma_row[w]/(float)sigma_row[v])*(1.0f+delta_row[v]);
              atomicAdd(delta_row[w], change);
            }
          }
        }
      }
      else
      {
        for (int kk = item.get_local_id(0) + endpoints_row[(current_depth)];
             kk < endpoints_row[current_depth + 1];
             kk += item.get_local_range(0))
        {
          int w = S_row[kk];
          float dsw = 0;
          float sw = (float)sigma_row[w];
          for(int z=R[w]; z<R[w+1]; z++)
          {
            int v = C[z];
            if(d_row[v] == (d_row[w]+1))
            {
              dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
            }
          }
          delta_row[w] = dsw;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
      if(j == 0)
      {
        current_depth--;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    for (int kk = item.get_local_id(0); kk < n; kk += item.get_local_range(0))
    {
      // Would need to check that kk != i here, but delta_row[kk] is guaranteed to be 0.
      atomicAdd(bc[kk], delta_row[kk]);
    }

    if(j == 0)
    {
      ind = atomicAdd(*next_source, 1);
      if(approx)
      {
        i = source_vertices[(ind)];
      }
      else
      {
        i = ind;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (ind == 2 * DIAMETER_SAMPLES)
    {

      for (int kk = item.get_local_id(0); kk < DIAMETER_SAMPLES;
           kk += item.get_local_range(0))
      {
        diameter_keys[kk] = diameters[kk];
      }
      item.barrier(sycl::access::fence_space::local_space);
      bitonic_sort(diameter_keys, DIAMETER_SAMPLES, item);
      item.barrier(sycl::access::fence_space::local_space);
      if(j == 0)
      {
        int log2n = 0;
        int tempn = n;
        while(tempn >>= 1)
        {
          ++log2n;
        }
        if(diameter_keys[DIAMETER_SAMPLES/2] < 4*log2n) //Use the median
        {
          *jia = 1;
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

std::vector<float> bc_gpu(
  sycl::queue &q,
  graph g,
  int max_threads_per_block,
  int number_of_SMs,
  program_options op,
  const std::set<int> &source_vertices)
{
  float *bc_gpu = new float[g.n];
  int next_source = number_of_SMs;

  size_t pitch_d, pitch_sigma, pitch_delta, pitch_Q, pitch_Q2, pitch_S, pitch_endpoints;

  const size_t dimGrid_x = number_of_SMs;
  sycl::range<1> gws (dimGrid_x * max_threads_per_block);
  sycl::range<1> lws (max_threads_per_block);

  float *bc_d = sycl::malloc_device<float>(g.n, q);
  q.memset(bc_d, 0, g.n * sizeof(float));

  int *R_d = sycl::malloc_device<int>(g.n+1, q);
  q.memcpy(R_d, g.R, sizeof(int) * (g.n + 1));

  int *C_d = sycl::malloc_device<int>(2*g.m, q);
  q.memcpy(C_d, g.C, sizeof(int) * (2*g.m));

  int *F_d = sycl::malloc_device<int>(2*g.m, q);
  q.memcpy(F_d, g.F, sizeof(int) * (2*g.m));

  #define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))

  pitch_d = PITCH_DEFAULT_ALIGN(sizeof(int) * g.n);
  int *d_d = (int*) sycl::malloc_device(pitch_d * dimGrid_x, q);

  pitch_sigma = PITCH_DEFAULT_ALIGN(sizeof(unsigned long long) * g.n);
  unsigned long long *sigma_d = (unsigned long long*) sycl::malloc_device(pitch_sigma * dimGrid_x, q);

  pitch_delta = PITCH_DEFAULT_ALIGN(sizeof(float) * g.n);
  float *delta_d = (float *) sycl::malloc_device(pitch_delta * dimGrid_x, q);

  pitch_Q = PITCH_DEFAULT_ALIGN(sizeof(int) * g.n);
  int *Q_d = (int*) sycl::malloc_device(pitch_Q * dimGrid_x, q);

  pitch_Q2 = PITCH_DEFAULT_ALIGN(sizeof(int) * g.n);
  int *Q2_d = (int*) sycl::malloc_device(pitch_Q2 * dimGrid_x, q);

  pitch_S = PITCH_DEFAULT_ALIGN(sizeof(int) * g.n);
  int *S_d = (int*) sycl::malloc_device(pitch_S * dimGrid_x, q);

  pitch_endpoints = PITCH_DEFAULT_ALIGN(sizeof(int) * (g.n+1));
  int *endpoints_d = (int*) sycl::malloc_device(pitch_endpoints * dimGrid_x, q);

  int *next_source_d = sycl::malloc_device<int>(1, q);
  q.memcpy(next_source_d, &next_source, sizeof(int));

  // source_vertices of type "std::set" has no data() method
  std::vector<int> source_vertices_h(source_vertices.size());
  std::copy(source_vertices.begin(),source_vertices.end(),source_vertices_h.begin());
  int *source_vertices_d = sycl::malloc_device<int>(source_vertices.size() + 1, q); // nonzeron buffer size

  if(op.approx)
  {
    q.memcpy(source_vertices_d, source_vertices_h.data(), source_vertices.size() * sizeof(int));
  }

  int *jia_d = sycl::malloc_device<int>(1, q);
  q.memset(jia_d, 0, sizeof(int));

  int *diameters_d = sycl::malloc_device<int>(DIAMETER_SAMPLES, q);
  q.memset(diameters_d, 0, sizeof(int));

  int end;
  bool approx;
  if(op.approx)
  {
    end = op.k;
    approx = true;
  } else {
    end = g.n;
    approx = false;
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&](sycl::handler &cgh) {
    auto g_n = g.n;
    auto g_m = g.m;
    sycl::local_accessor<int, 0> ind_sm (cgh);
    sycl::local_accessor<int, 0> i_sm (cgh);
    sycl::local_accessor<int*, 0> Q_row_sm (cgh);
    sycl::local_accessor<int*, 0> Q2_row_sm (cgh);
    sycl::local_accessor<int*, 0> S_row_sm (cgh);
    sycl::local_accessor<int*, 0> endpoints_row_sm (cgh);
    sycl::local_accessor<int, 0> Q_len_sm (cgh);
    sycl::local_accessor<int, 0> Q2_len_sm (cgh);
    sycl::local_accessor<int, 0> S_len_sm (cgh);
    sycl::local_accessor<int, 0> current_depth_sm (cgh);
    sycl::local_accessor<int, 0> endpoints_len_sm (cgh);
    sycl::local_accessor<bool, 0> sp_calc_done_sm (cgh);
    sycl::local_accessor<int, 0> next_index_sm (cgh);
    sycl::local_accessor<int, 1> diameter_keys_sm (sycl::range<1>(DIAMETER_SAMPLES), cgh);

    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      bc_kernel(
        bc_d,
        R_d,
        C_d,
        F_d,
        g_n, g_m,
        d_d,
        sigma_d,
        delta_d,
        Q_d,
        Q2_d,
        S_d,
        endpoints_d,
        next_source_d,
        pitch_d,
        pitch_sigma,
        pitch_delta,
        pitch_Q,
        pitch_Q2,
        pitch_S,
        pitch_endpoints,
        0, end,
        jia_d,
        diameters_d,
        source_vertices_d,
        approx,
        item,
        ind_sm, i_sm,
        Q_row_sm, Q2_row_sm,
        S_row_sm, endpoints_row_sm,
        Q_len_sm, Q2_len_sm,
        S_len_sm, current_depth_sm,
        endpoints_len_sm,
        sp_calc_done_sm,
        next_index_sm,
        diameter_keys_sm.get_pointer());
     });
  }).wait();

  auto stop = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
  std::cout << "Kernel execution time " << time * 1e-9f << " (s)\n";

  // GPU result
  q.memcpy(bc_gpu, bc_d, g.n * sizeof(float)).wait();

  sycl::free(bc_d, q);
  sycl::free(R_d, q);
  sycl::free(C_d, q);
  sycl::free(F_d, q);
  sycl::free(d_d, q);
  sycl::free(sigma_d, q);
  sycl::free(delta_d, q);
  sycl::free(Q_d, q);
  sycl::free(Q2_d, q);
  sycl::free(S_d, q);
  sycl::free(endpoints_d, q);
  sycl::free(next_source_d, q);
  sycl::free(jia_d, q);
  sycl::free(diameters_d, q);
  sycl::free(source_vertices_d, q);

  //Copy GPU result to a vector
  std::vector<float> bc_gpu_v(bc_gpu,bc_gpu+g.n);

  for(int i=0; i<g.n; i++)
  {
    bc_gpu_v[i] /= 2.0f; //we don't want to double count the unweighted edges
  }

  delete[] bc_gpu;
  return bc_gpu_v;
}

// query the properties of a single device for simplicity
void query_device(sycl::queue &q, int &max_threads_per_block, int &number_of_SMs, program_options op)
{
  auto dev = q.get_device();
  std::cout << "Chosen Device: " << dev.get_info<sycl::info::device::name>() << std::endl;

  max_threads_per_block = dev.get_info<sycl::info::device::max_work_group_size>();
  number_of_SMs = dev.get_info<sycl::info::device::max_compute_units>();

  std::cout << "Number of Multiprocessors: " << number_of_SMs << std::endl;
  std::cout << "Size of Global Memory: "
            << dev.get_info<sycl::info::device::global_mem_size>() / (float)(1024 * 1024 * 1024) << " GB"
            << std::endl << std::endl;
}

