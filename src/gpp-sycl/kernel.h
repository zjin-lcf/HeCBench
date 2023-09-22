template<typename T, sycl::memory_scope MemoryScope = sycl::memory_scope::device>
static inline void atomicFetchAdd(T& val, const T delta)
{
  sycl::atomic_ref<T, 
     sycl::memory_order::relaxed, 
     MemoryScope, sycl::access::address_space::global_space> ref(val);
  ref.fetch_add(delta);
}

void solver(
    sycl::nd_item<3> &item,
    int number_bands, int ngpown, int ncouls,
    const int *__restrict inv_igp_index,
    const int *__restrict indinv,
    const dataType *__restrict wx_array,
    const CustomComplex<dataType> *__restrict wtilde_array,
    const CustomComplex<dataType> *__restrict aqsmtemp,
    const CustomComplex<dataType> *__restrict aqsntemp,
    const CustomComplex<dataType> *__restrict I_eps_array,
    const dataType *__restrict vcoul,
    dataType *__restrict achtemp_re,
    dataType *__restrict achtemp_im) 
{
  dataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
  for (int iw = nstart; iw < nend; ++iw) {
    achtemp_re_loc[iw] = 0.0;
    achtemp_im_loc[iw] = 0.0;
  }

  const int threadIdx_x = item.get_local_id(2);
  const int blockDim_x = item.get_local_range(2);
  const int blockIdx_x = item.get_group(2);
  const int blockIdx_y = item.get_group(1);
  const int gridDim_x = item.get_group_range(2);
  const int gridDim_y = item.get_group_range(1);

  for (int n1 = blockIdx_x; n1 < number_bands; n1 += gridDim_x) // 512 iterations
  {
    for (int my_igp = blockIdx_y; my_igp < ngpown; my_igp += gridDim_y) // 1634 iterations
    {
      int indigp = inv_igp_index[my_igp];
      int igp = indinv[indigp];
      CustomComplex<dataType> sch_store1 =
          CustomComplex_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 *
          vcoul[igp];

      for (int ig = threadIdx_x; ig < ncouls; ig += blockDim_x) {
        #pragma unroll
        for (int iw = nstart; iw < nend; ++iw) // 3 iterations
        {
          CustomComplex<dataType> wdiff =
              wx_array[iw] - wtilde_array(my_igp, ig);
          CustomComplex<dataType> delw =
              wtilde_array(my_igp, ig) * CustomComplex_conj(wdiff) *
              (1 / CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
          CustomComplex<dataType> sch_array =
              delw * I_eps_array(my_igp, ig) * sch_store1;

          achtemp_re_loc[iw] += CustomComplex_real(sch_array);
          achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
        }
      }
    } // ngpown
  }   // number_bands

  // Add the final results here
  for (int iw = nstart; iw < nend; ++iw) {
    atomicFetchAdd(achtemp_re[iw], achtemp_re_loc[iw]);
    atomicFetchAdd(achtemp_im[iw], achtemp_im_loc[iw]);
  }
}
