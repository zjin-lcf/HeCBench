/***********************************************
  streamcluster_cl.h
  : parallelized code of streamcluster

  - original code from PARSEC Benchmark Suite
  - parallelization with OpenCL API has been applied by
  Jianbin Fang - j.fang@tudelft.nl
  Delft University of Technology
  Faculty of Electrical Engineering, Mathematics and Computer Science
  Department of Software Technology
  Parallel and Distributed Systems Group
  on 15/03/2010
 ***********************************************/

#define THREADS_PER_BLOCK 256
#define MAXBLOCKS 65536
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

//#define PROFILE_TMP
typedef struct dpct_type_ba53f5 {
  float weight;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point_Struct;


// CUDA kernel 
#include "kernel.h"

/* host memory analogous to device memory. These memories are allocated in the function,
 * but they are freed in the streamcluster.cpp. We cannot free them in the function as
 * the funtion is called repeatedly in streamcluster.cpp. */
float *work_mem_h;
float *coord_h;
float *gl_lower;
Point_Struct *p_h;

// device memory
float *work_mem_d;
float *coord_d;
int   *center_table_d;
char  *switch_membership_d;
Point_Struct *p_d;

static int c;      // counters

void quit(char *message){
  printf("%s\n", message);
  exit(1);
}
float pgain( long x, Points *points, float z, long int *numcenters, 
    int kmax, bool *is_center, int *center_table, char *switch_membership,
    double *serial, double *cpu_gpu_memcpy, double *memcpy_back, double *gpu_malloc, double *kernel_time) {
  float gl_cost = 0;
  try{
#ifdef PROFILE_TMP
    double t1 = gettime();
#endif
    int K  = *numcenters ;            // number of centers
    int num    =   points->num;        // number of points
    int dim     =   points->dim;        // number of dimension
    kmax++;
    /***** build center index table 1*****/
    int count = 0;
    for( int i=0; i<num; i++){
      if( is_center[i] )
        center_table[i] = count++;
    }

#ifdef PROFILE_TMP
    double t2 = gettime();
    *serial += t2 - t1;
#endif

    /***** initial memory allocation and preparation for transfer : execute once -1 *****/
    if( c == 0 ) {
#ifdef PROFILE_TMP
      double t3 = gettime();
#endif
      coord_h = (float*) malloc( num * dim * sizeof(float));                // coordinates (host)
      gl_lower = (float*) malloc( kmax * sizeof(float) );
      work_mem_h = (float*) malloc (kmax*num*sizeof(float));
      p_h = (Point_Struct*)malloc(num*sizeof(Point_Struct));  //by cambine: not compatibal with original Point

      // prepare mapping for point coordinates
      //--cambine: what's the use of point coordinates? for computing distance.
      for(int i=0; i<dim; i++){
        for(int j=0; j<num; j++)
          coord_h[ (num*i)+j ] = points->p[j].coord[i];
      }
#ifdef PROFILE_TMP    
      double t4 = gettime();
      *serial += t4 - t3;
#endif
      dpct::dpct_malloc((void **)&work_mem_d, sizeof(float) * (kmax + 1) * num);
      dpct::dpct_malloc((void **)&center_table_d, sizeof(int) * num);
      dpct::dpct_malloc((void **)&switch_membership_d, sizeof(char) * num);
      dpct::dpct_malloc((void **)&coord_d, sizeof(float) * dim * num);
      dpct::dpct_malloc((void **)&p_d, sizeof(Point_Struct) * num);

#ifdef PROFILE_TMP
      double t5 = gettime();
      *gpu_malloc += t5 - t4;
#endif

      // copy coordinate to device memory
      dpct::async_dpct_memcpy(coord_d, coord_h, sizeof(float) * num * dim,
                              dpct::host_to_device);
#ifdef PROFILE_TMP
      cudaDeviceSynchronize();
      double t6 = gettime();
      *cpu_gpu_memcpy += t6 - t4;
#endif
    }    // first iteration

#ifdef PROFILE_TMP
    double t100 = gettime();
#endif

    for(int i=0; i<num; i++){
      p_h[i].weight = ((points->p)[i]).weight;
      p_h[i].assign = ((points->p)[i]).assign;
      p_h[i].cost = ((points->p)[i]).cost;  
    }

#ifdef PROFILE_TMP
    double t101 = gettime();
    *serial += t101 - t100;
#endif
#ifdef PROFILE_TMP
    double t7 = gettime();
#endif
    /***** memory transfer from host to device *****/
    dpct::async_dpct_memcpy(center_table_d, center_table, sizeof(int) * num,
                            dpct::host_to_device);
    dpct::async_dpct_memcpy(p_d, p_h, sizeof(Point_Struct) * num,
                            dpct::host_to_device);
#ifdef PROFILE_TMP
    cudaDeviceSynchronize();
    double t8 = gettime();
    *cpu_gpu_memcpy += t8 - t7;
#endif

    /***** kernel execution *****/
    /* Determine the number of thread blocks in the x- and y-dimension */
    size_t smSize = dim; // * sizeof(float);

    // reset on the host
    //::memset(switch_membership, 0, num);

#ifdef PROFILE_TMP
    double t9 = gettime();
#endif

    dpct::async_dpct_memset(switch_membership_d, 0, num * sizeof(char));

    dpct::async_dpct_memset(work_mem_d, 0, num * (K + 1) * sizeof(float));

    int work_group_size = THREADS_PER_BLOCK;
    int work_items = num;
    if(work_items%work_group_size != 0)  //process situations that work_items cannot be divided by work_group_size
      work_items = work_items + (work_group_size-(work_items%work_group_size));

    {
      std::pair<dpct::buffer_t, size_t> p_d_buf_ct0 =
          dpct::get_buffer_and_offset(p_d);
      size_t p_d_offset_ct0 = p_d_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> coord_d_buf_ct1 =
          dpct::get_buffer_and_offset(coord_d);
      size_t coord_d_offset_ct1 = coord_d_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> work_mem_d_buf_ct2 =
          dpct::get_buffer_and_offset(work_mem_d);
      size_t work_mem_d_offset_ct2 = work_mem_d_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> center_table_d_buf_ct3 =
          dpct::get_buffer_and_offset(center_table_d);
      size_t center_table_d_offset_ct3 = center_table_d_buf_ct3.second;
      std::pair<dpct::buffer_t, size_t> switch_membership_d_buf_ct4 =
          dpct::get_buffer_and_offset(switch_membership_d);
      size_t switch_membership_d_offset_ct4 =
          switch_membership_d_buf_ct4.second;
      dpct::get_default_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(smSize * sizeof(float)), cgh);
        auto p_d_acc_ct0 =
            p_d_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
        auto coord_d_acc_ct1 =
            coord_d_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto work_mem_d_acc_ct2 =
            work_mem_d_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto center_table_d_acc_ct3 =
            center_table_d_buf_ct3.first
                .get_access<sycl::access::mode::read_write>(cgh);
        auto switch_membership_d_acc_ct4 =
            switch_membership_d_buf_ct4.first
                .get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, work_items / work_group_size) *
                    sycl::range<3>(1, 1, work_group_size),
                sycl::range<3>(1, 1, work_group_size)),
            [=](sycl::nd_item<3> item_ct1) {
              const Point_Struct *p_d_ct0 =
                  (const Point_Struct *)(&p_d_acc_ct0[0] + p_d_offset_ct0);
              const float *coord_d_ct1 =
                  (const float *)(&coord_d_acc_ct1[0] + coord_d_offset_ct1);
              float *work_mem_d_ct2 =
                  (float *)(&work_mem_d_acc_ct2[0] + work_mem_d_offset_ct2);
              const int *center_table_d_ct3 =
                  (const int *)(&center_table_d_acc_ct3[0] +
                                center_table_d_offset_ct3);
              char *switch_membership_d_ct4 =
                  (char *)(&switch_membership_d_acc_ct4[0] +
                           switch_membership_d_offset_ct4);
              compute_cost(p_d_ct0, coord_d_ct1, work_mem_d_ct2,
                           center_table_d_ct3, switch_membership_d_ct4, num,
                           dim, x, K, item_ct1,
                           dpct_local_acc_ct1.get_pointer());
            });
      });
    }

#ifdef PROFILE_TMP
    cudaDeviceSynchronize();
    double t10 = gettime();
    *kernel_time += t10 - t9;
#endif

    /***** copy back to host for CPU side work *****/
    dpct::dpct_memcpy(switch_membership, switch_membership_d,
                      num * sizeof(char), dpct::device_to_host);

    // reset on the host already
    dpct::dpct_memcpy(work_mem_h, work_mem_d, num * (K + 1) * sizeof(float),
                      dpct::device_to_host);

#ifdef PROFILE_TMP
    double t11 = gettime();
    *memcpy_back += t11 - t10;
#endif

    /****** cpu side work *****/
    int numclose = 0;
    gl_cost = z;

    /* compute the number of centers to close if we are to open i */
    for(int i=0; i < num; i++){  //--cambine:??
      if( is_center[i] ) {
        float low = z;
        //printf("i=%d  ", i);
        for( int j = 0; j < num; j++ )
          low += work_mem_h[ j*(K+1) + center_table[i] ];
        //printf("low=%f\n", low);    
        gl_lower[center_table[i]] = low;

        if ( low > 0 ) {
          numclose++;        
          work_mem_h[i*(K+1)+K] -= low;
        }
      }
      gl_cost += work_mem_h[i*(K+1)+K];
    }

    /* if opening a center at x saves cost (i.e. cost is negative) do so
       otherwise, do nothing */
    if ( gl_cost < 0 ) {
      for(int i=0; i<num; i++){

        bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
        if ( (switch_membership[i]=='1') || close_center ) {
          points->p[i].cost = points->p[i].weight * dist(points->p[i], points->p[x], points->dim);
          points->p[i].assign = x;
        }
      }

      for(int i=0; i<num; i++){
        if( is_center[i] && gl_lower[center_table[i]] > 0 )
          is_center[i] = false;
      }

      is_center[x] = true;
      *numcenters = *numcenters +1 - numclose;
    }
    else
      gl_cost = 0;  // the value we'

#ifdef PROFILE_TMP
    double t12 = gettime();
    *serial += t12 - t11;
#endif
    c++;
  }
  catch(string msg){
    printf("--cambine:%s\n", msg.c_str());
    exit(-1);    
  }
  catch(...){
    printf("--cambine: unknow reasons in pgain\n");
  }

#ifdef DEBUG
  FILE *fp = fopen("data_opencl.txt", "a");
  fprintf(fp,"%d, %f\n", c, gl_cost);
  fclose(fp);
#endif
  return -gl_cost;
}
