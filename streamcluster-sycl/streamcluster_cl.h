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

typedef struct {
  float weight;
  long assign;  /* number of point where this one is assigned */
  float cost;  /* cost of that assignment, weight*distance */
} Point_Struct;

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

float pgain( long x, Points *points, float z, long int *numcenters,
             int kmax, bool *is_center, int *center_table, char *switch_membership,
             double *serial, double *cpu_gpu_memcpy, double *memcpy_back,
             double *gpu_malloc, double *kernel_time) {

  float gl_cost = 0;
  try{
#ifdef PROFILE_TMP
    double t1 = gettime();
#endif
    int K = *numcenters;   // number of centers
    int num = points->num; // number of points
    int dim = points->dim; // number of dimensions
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

    /***** initial memory allocation and preparation for transfer : execute once *****/
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

      work_mem_d = sycl::malloc_device<float>((kmax+1) * num, q);
      center_table_d = sycl::malloc_device<int> (num, q);
      switch_membership_d = sycl::malloc_device<char>(num, q);
      p_d = sycl::malloc_device<Point_Struct>(num, q);
      coord_d = sycl::malloc_device<float> (dim * num, q);

#ifdef PROFILE_TMP
      double t5 = gettime();
      *gpu_malloc += t5 - t4;
#endif

      // copy coordinate to device memory
      q.memcpy(coord_d, coord_h, sizeof(float)*num*dim);

#ifdef PROFILE_TMP
      q.wait();
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
    q.memcpy(center_table_d, center_table, sizeof(int)*num);
    q.memcpy(p_d, p_h, sizeof(Point_Struct)*num);

#ifdef PROFILE_TMP
    q.wait();
    double t8 = gettime();
    *cpu_gpu_memcpy += t8 - t7;
#endif

    /***** kernel execution *****/
    /* Determine the number of thread blocks in the x- and y-dimension */
    size_t smSize = dim; // * sizeof(float);

#ifdef PROFILE_TMP
    double t9 = gettime();
#endif

    q.memset(switch_membership_d, 0, num * sizeof(char));
    q.memset(work_mem_d, 0, num*(K+1)*sizeof(float));

    int work_group_size = THREADS_PER_BLOCK;
    int work_items = num;

    //process situations that work_items cannot be divided by work_group_size
    if(work_items%work_group_size != 0)
      work_items = work_items + (work_group_size-(work_items%work_group_size));

    q.submit([&](sycl::handler& cgh) {
      auto work_mem_d_acc = work_mem_d;
      auto coord_d_acc = coord_d;
      auto center_table_d_acc = center_table_d;
      auto switch_membership_d_acc = switch_membership_d;
      auto p_d_acc = p_d;
      sycl::local_accessor <float, 1> coord_s_acc (sycl::range<1>(smSize), cgh);
      cgh.parallel_for<class kernel_pgain>(
        sycl::nd_range<1>(sycl::range<1>(work_items),
                          sycl::range<1>(work_group_size)), [=] (sycl::nd_item<1> item) {
          #include "kernel.sycl"
      });
    });

#ifdef PROFILE_TMP
    q.wait();
    double t10 = gettime();
    *kernel_time += t10 - t9;
#endif

    /***** copy back to host for CPU side work *****/
    q.memcpy(switch_membership, switch_membership_d, num * sizeof(char));
    q.memcpy(work_mem_h, work_mem_d, num*(K+1) * sizeof(float));
    q.wait();

#ifdef PROFILE_TMP
    double t11 = gettime();
    *memcpy_back += t11 - t10;
#endif

    /****** cpu side work *****/
    int numclose = 0;
    gl_cost = z;

    /* compute the number of centers to close if we are to open i */
    for(int i=0; i < num; i++){
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
      gl_cost = 0;

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
  FILE *fp = fopen("data_debug.txt", "a");
  fprintf(fp,"%d, %f\n", c, gl_cost);
  fclose(fp);
#endif
  return -gl_cost;
}
