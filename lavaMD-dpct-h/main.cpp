#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
// (in path known to compiler)      needed by printf
// (in path known to compiler)      needed by malloc
#include <stdbool.h>        // (in path known to compiler)      needed by true/false
#include "./util/timer/timer.h"    // (in path specified here)
#include "./util/num/num.h"        // (in path specified here)
#include "./main.h"            // (in the current directory)

// CUDA kernel
#include "kernel.h"

int main(  int argc, char *argv [])
{

  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str* box_cpu;
  FOUR_VECTOR* rv_cpu;
  fp* qv_cpu;
  FOUR_VECTOR* fv_cpu;
  int nh;

  printf("WG size of kernel = %d \n", NUMBER_THREADS);

  // assing default values
  dim_cpu.arch_arg = 0;
  dim_cpu.cores_arg = 1;
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  if(argc==3){
    for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
      // check if -boxes1d
      if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
        // check if value provided
        if(argc>=dim_cpu.cur_arg+1){
          // check if value is a number
          if(isInteger(argv[dim_cpu.cur_arg+1])==1){
            dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
            if(dim_cpu.boxes1d_arg<0){
              printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
              return 0;
            }
            dim_cpu.cur_arg = dim_cpu.cur_arg+1;
          }
          // value is not a number
          else{
            printf("ERROR: Value to -boxes1d argument in not a number\n");
            return 0;
          }
        }
        // value not provided
        else{
          printf("ERROR: Missing value to -boxes1d argument\n");
          return 0;
        }
      }
      // unknown
      else{
        printf("ERROR: Unknown argument\n");
        return 0;
      }
    }
    // Print configuration
    printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n", dim_cpu.arch_arg, dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
  }
  else{
    printf("Provide boxes1d argument, example: -boxes1d 16");
    return 0;
  }

  par_cpu.alpha = 0.5;

  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; 

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  // allocate boxes
  box_cpu = (box_str*)malloc(dim_cpu.box_mem);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for(i=0; i<dim_cpu.boxes1d_arg; i++){
    // home boxes in y direction
    for(j=0; j<dim_cpu.boxes1d_arg; j++){
      // home boxes in x direction
      for(k=0; k<dim_cpu.boxes1d_arg; k++){

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for(l=-1; l<2; l++){
          // neighbor boxes in y direction
          for(m=-1; m<2; m++){
            // neighbor boxes in x direction
            for(n=-1; n<2; n++){

              // check if (this neighbor exists) and (it is not the same as home box)
              if(    (((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)  &&
                  (l==0 && m==0 && n==0)==false  ){

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
                box_cpu[nh].nei[box_cpu[nh].nn].number =  (box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
                  (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
                  box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } // neighbor boxes in x direction
          } // neighbor boxes in y direction
        } // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    } // home boxes in y direction
  } // home boxes in z direction

  //  PARAMETERS, DISTANCE, CHARGE AND FORCE

  // random generator seed set to random value - time in this case
  srand(2);

  // input (distances)
  rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    rv_cpu[i].v = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].x = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].y = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
    rv_cpu[i].z = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
  }

  // input (charge)
  qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    qv_cpu[i] = (rand()%10 + 1) / 10.0;      // get a number in the range 0.1 - 1.0
  }

  // output (forces)
  fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fv_cpu[i].v = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].x = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].y = 0;                // set to 0, because kernels keeps adding to initial value
    fv_cpu[i].z = 0;                // set to 0, because kernels keeps adding to initial value
  }

  long long start = get_time();

  int dim_cpu_number_boxes = dim_cpu.number_boxes;

  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

  box_str* d_box_gpu;
  FOUR_VECTOR* d_rv_gpu;
  fp* d_qv_gpu;
  FOUR_VECTOR* d_fv_gpu;

 dpct::dpct_malloc((void **)&d_box_gpu, dim_cpu.box_mem);
 dpct::dpct_malloc((void **)&d_rv_gpu, dim_cpu.space_mem);
 dpct::dpct_malloc((void **)&d_qv_gpu, dim_cpu.space_mem2);
 dpct::dpct_malloc((void **)&d_fv_gpu, dim_cpu.space_mem);

 dpct::dpct_memcpy(d_box_gpu, box_cpu, dim_cpu.box_mem, dpct::host_to_device);
 dpct::dpct_memcpy(d_rv_gpu, rv_cpu, dim_cpu.space_mem, dpct::host_to_device);
 dpct::dpct_memcpy(d_qv_gpu, qv_cpu, dim_cpu.space_mem2, dpct::host_to_device);
 dpct::dpct_memcpy(d_fv_gpu, fv_cpu, dim_cpu.space_mem, dpct::host_to_device);

 {
  dpct::buffer_t d_box_gpu_buf_ct0 = dpct::get_buffer(d_box_gpu);
  dpct::buffer_t d_rv_gpu_buf_ct1 = dpct::get_buffer(d_rv_gpu);
  dpct::buffer_t d_qv_gpu_buf_ct2 = dpct::get_buffer(d_qv_gpu);
  dpct::buffer_t d_fv_gpu_buf_ct3 = dpct::get_buffer(d_fv_gpu);
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
   sycl::accessor<FOUR_VECTOR, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       rA_shared_acc_ct1(sycl::range<1>(100), cgh);
   sycl::accessor<FOUR_VECTOR, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       rB_shared_acc_ct1(sycl::range<1>(100), cgh);
   sycl::accessor<fp, 1, sycl::access::mode::read_write,
                  sycl::access::target::local>
       qB_shared_acc_ct1(sycl::range<1>(100), cgh);
   auto d_box_gpu_acc_ct0 =
       d_box_gpu_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
   auto d_rv_gpu_acc_ct1 =
       d_rv_gpu_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
   auto d_qv_gpu_acc_ct2 =
       d_qv_gpu_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
   auto d_fv_gpu_acc_ct3 =
       d_fv_gpu_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);

   cgh.parallel_for(
       sycl::nd_range<3>(sycl::range<3>(1, 1, dim_cpu_number_boxes) *
                             sycl::range<3>(1, 1, NUMBER_THREADS),
                         sycl::range<3>(1, 1, NUMBER_THREADS)),
       [=](sycl::nd_item<3> item_ct1) {
        md((const box_str *)(&d_box_gpu_acc_ct0[0]),
           (const FOUR_VECTOR *)(&d_rv_gpu_acc_ct1[0]),
           (const float *)(&d_qv_gpu_acc_ct2[0]),
           (FOUR_VECTOR *)(&d_fv_gpu_acc_ct3[0]), par_cpu.alpha,
           dim_cpu_number_boxes, item_ct1, rA_shared_acc_ct1.get_pointer(),
           rB_shared_acc_ct1.get_pointer(), qB_shared_acc_ct1.get_pointer());
       });
  });
 }

 dpct::dpct_memcpy(fv_cpu, d_fv_gpu, dim_cpu.space_mem, dpct::device_to_host);

 dpct::dpct_free(d_box_gpu);
 dpct::dpct_free(d_rv_gpu);
 dpct::dpct_free(d_qv_gpu);
 dpct::dpct_free(d_fv_gpu);

  long long end = get_time();
  printf("Device offloading time:\n"); 
  printf("%.12f s\n", (float) (end-start) / 1000000); 

  // dump results
#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("result.txt", "w");  
  for(i=0; i<dim_cpu.space_elem; i=i+1){
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
  }
  fclose(fptr);
#endif         


  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);

  return 0; 

}


