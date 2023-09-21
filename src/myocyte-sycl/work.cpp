#include <sycl/sycl.hpp>
#include "kernel_ecc.cpp"
#include "kernel_cam.cpp"
#include "kernel_fin.cpp"
#include "kernel.cpp"
#include "master.cpp"
#include "embedded_fehlberg_7_8.cpp"
#include "solver.cpp"

int work(int xmax, int workload)
{
  //============================================================
  //		VARIABLE
  //============================================================

  //============================================================
  //		TIME
  //============================================================

  auto time0 = std::chrono::steady_clock::now();

  //===========================================================
  //		COUNTERS
  //===========================================================

  long long memory;
  int i,j;
  int status;

  //==========================================================
  //		DATA
  //==========================================================

  fp*** y;
  fp** x;
  fp** params;
  fp* com;

  auto time1 = std::chrono::steady_clock::now();

  //==========================================================
  // 	ALLOCATE MEMOR
  //==========================================================

  //==========================================================
  //		MEMORY CHECK
  //==========================================================

  memory = workload*(xmax+1)*EQUATIONS*4;
  if(memory>1000000000){
    printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
    return 0;
  }

  //=========================================================
  // 	ALLOCATE ARRAYS
  //=========================================================

  y = (fp ***) malloc(workload* sizeof(fp **));
  for(i=0; i<workload; i++){
    y[i] = (fp**)malloc((1+xmax)*sizeof(fp*));
    for(j=0; j<(1+xmax); j++){
      y[i][j]= (fp *) malloc(EQUATIONS* sizeof(fp));
    }
  }

  x = (fp **) malloc(workload * sizeof(fp *));
  for (i= 0; i<workload; i++){
    x[i]= (fp *)malloc((1+xmax) *sizeof(fp));
  }

  params = (fp **) malloc(workload * sizeof(fp *));
  for (i= 0; i<workload; i++){
    params[i]= (fp *)malloc(PARAMETERS * sizeof(fp));
  }

  com = (fp*)malloc(3 * sizeof(fp));

  //=========================================================
  // 	ALLOCATE ARRAYS
  //=========================================================
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int d_initvalu_mem;
  d_initvalu_mem = EQUATIONS * sizeof(fp);
  fp* d_initvalu;
  d_initvalu = (float *)sycl::malloc_device(d_initvalu_mem, q);

  int d_finavalu_mem;
  d_finavalu_mem = EQUATIONS * sizeof(fp);
  fp* d_finavalu;
  d_finavalu = (float *)sycl::malloc_device(d_finavalu_mem, q);

  int d_params_mem;
  d_params_mem = PARAMETERS * sizeof(fp);
  fp* d_params;
  d_params = (float *)sycl::malloc_device(d_params_mem, q);

  int d_com_mem;
  d_com_mem = 3 * sizeof(fp);
  fp* d_com;
  d_com = (float *)sycl::malloc_device(d_com_mem, q);

  auto time2 = std::chrono::steady_clock::now();

  //==========================================================
  // 	INITIAL VALUES
  //==========================================================

  // y
  for(i=0; i<workload; i++){
    read("../data/myocyte/y.txt", y[i][0], EQUATIONS, 1, 0);
  }

  // params
  for(i=0; i<workload; i++){
    read("../data/myocyte/params.txt", params[i], PARAMETERS, 1, 0);
  }

  auto time3 = std::chrono::steady_clock::now();

  //==========================================================
  //	EXECUTION
  //==========================================================

  for(i=0; i<workload; i++){

    status = solver(
        q,
        y[i],
        x[i],
        xmax,
        params[i],
        com,
        d_initvalu,
        d_finavalu,
        d_params,
        d_com);

    if(status !=0){
      printf("STATUS: %d\n", status);
    }
  }

  // print results to output.txt
  FILE * pFile;
  pFile = fopen ("output.txt","w");
  if (pFile==NULL)
  {
    fprintf (stderr, "ERROR: failed to open output.txt for writing.\n");
    return -1;
  }

  int k;
  for(i=0; i<workload; i++){
    fprintf(pFile, "WORKLOAD %d:\n", i);
    for(j=0; j<(xmax+1); j++){
      fprintf(pFile, "\tTIME %d:\n", j);
      for(k=0; k<EQUATIONS; k++){
        fprintf(pFile, "\t\ty[%d][%d][%d]=%10.7e\n", i, j, k, y[i][j][k]);
      }
    }
  }

  fclose (pFile);

  auto time4 = std::chrono::steady_clock::now();

  //========================================================
  //	DEALLOCATION
  //========================================================

  // y values
  for (i= 0; i< workload; i++){
    for (j= 0; j< (1+xmax); j++){
      free(y[i][j]);
    }
    free(y[i]);
  }
  free(y);

  // x values
  for (i= 0; i< workload; i++){
    free(x[i]);
  }
  free(x);

  // parameters
  for (i= 0; i< workload; i++){
    free(params[i]);
  }
  free(params);

  auto time5 = std::chrono::steady_clock::now();

  // com
  free(com);

  // GPU memory
  sycl::free(d_initvalu, q);
  sycl::free(d_finavalu, q);
  sycl::free(d_params, q);
  sycl::free(d_com, q);

  // DISPLAY TIMING

  auto etime1 = std::chrono::duration_cast<std::chrono::nanoseconds>(time1 - time0).count();
  auto etime2 = std::chrono::duration_cast<std::chrono::nanoseconds>(time2 - time1).count();
  auto etime3 = std::chrono::duration_cast<std::chrono::nanoseconds>(time3 - time2).count();
  auto etime4 = std::chrono::duration_cast<std::chrono::nanoseconds>(time4 - time3).count();
  auto etime5 = std::chrono::duration_cast<std::chrono::nanoseconds>(time5 - time4).count();
  auto etime6 = std::chrono::duration_cast<std::chrono::nanoseconds>(time5 - time0).count();

  printf("Time spent in different stages of the application:\n");
  printf("%.12f s, %.12f %% : SETUP VARIABLES\n", 
      etime1 * 1e-9, (float) (etime1) / (float) (etime6) * 100);
  printf("%.12f s, %.12f %% : ALLOCATE CPU MEMORY AND GPU MEMORY\n",
      etime2 * 1e-9, (float) (etime2) / (float) (etime6) * 100);
  printf("%.12f s, %.12f %% : READ DATA FROM FILES\n",
      etime3 * 1e-9, (float) (etime3) / (float) (etime6) * 100);
  printf("%.12f s, %.12f %% : RUN COMPUTATION\n",
      etime4 * 1e-9, (float) (etime4) / (float) (etime6) * 100);
  printf("%.12f s, %.12f %% : FREE MEMORY\n",
      etime5 * 1e-9, (float) (etime5) / (float) (etime6) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", etime6 * 1e-9);

  return 0;
}
