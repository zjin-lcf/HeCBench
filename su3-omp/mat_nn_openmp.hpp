// OpenMP target offload implementation
#include <omp.h>
#include <unistd.h>

#define THREADS_PER_SITE 36

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
              size_t total_sites, size_t iterations, size_t threads_per_team, int use_device)
{
  if (threads_per_team == 0)
    threads_per_team = THREADS_PER_SITE;

  site *d_a, *d_c;
  su3_matrix *d_b;
  d_a = a.data(); 
  d_b = b.data();
  d_c = c.data();
 
  // Move A and B data to the device, Allocate C data
  double ttotal;
  #pragma omp target data map(to: d_a[0:total_sites], d_b[0:4]) map(from: d_c[0:total_sites])
  {  // begin OpenMP block

  // This code improves performance over above baseline
  // Similar to Cuda and OpenCL work item approach
  // Initial contribution by Xinmin Tian, Intel
  size_t num_work_items = total_sites * threads_per_team; 

  if (verbose >= 1) {
    std::cout << "Number of teams = " << total_sites << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
    std::cout << "Number of work items = " << num_work_items << std::endl;
  }

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      tstart = Clock::now();
    }
    #pragma omp target teams distribute parallel for num_teams(total_sites) thread_limit(threads_per_team)
    for (int id =0; id < num_work_items; id++) {
      int i = id/36;
      if (i < total_sites) {
        int j = (id%36)/9;
        int k = (id%9)/3;
        int l = id%3;

        Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
        for(int m=0;m<3;m++) {
          cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
        }
        d_c[i].link[j].e[k][l] = cc;
#else
        for(int m=0;m<3;m++) {
           CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
        }
        d_c[i].link[j].e[k][l].real = cc.real;
        d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
      }
    }
  }

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  } // end of OpenMP block, C gets moved back to the host

  // It is not possible to check for NaNs when the application is compiled with -ffast-math
  // Therefore we print out the calculated checksum as a manual check for the user.
  // This is helpful when using LLVM/Clang-10.0 to compile the OpenMP target offload
  // implementation without MILC_COMPLEX (i.e. using std::complex).
  double sum = 0.0;
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j)  for(int k=0;k<3;++k)  for(int l=0;l<3;++l) {
    Complx cc = {0.0, 0.0};
    for(int m=0;m<3;m++) {
      #ifdef MILC_COMPLEX
        CMULSUM( a[i].link[j].e[k][m], b[j].e[m][l], cc)
      #else
        cc += a[i].link[j].e[k][m] * b[j].e[m][l];
      #endif
    }

    #ifdef MILC_COMPLEX
      sum += c[i].link[j].e[k][l].real;
    #else
      sum += std::real(c[i].link[j].e[k][l]);
    #endif
  }
  sum /= (double)total_sites;
  if (almost_equal(sum, 4.0*sizeof(su3_matrix)/(sizeof(Complx)), 1E-6)) {
    printf("Checksum SUCCESS... though please be diligent and check the "
    "following value is not NaN: checksum=%.0lf\n", sum);
  } else {
    printf("Checksum FAILURE\n");
  }

  return (ttotal /= 1.0e6);
}
