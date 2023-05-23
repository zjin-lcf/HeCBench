#include <chrono>
#include <sycl/sycl.hpp>
#include "hotspot.h"

// Returns the current system time in microseconds
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {
  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 ) {
    printf( "Unable to open file %s\n", file );
    return;
  }

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++)
    {
      sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
    }

  fclose(fp);	
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 ) {
    printf( "The file %s was not opened successfully", file );
    exit(-1);
  }

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) {
        printf("Error reading file\n");
        exit(-1);
      }
      if (feof(fp)) {
        printf("not enough lines in file");
        exit(-1);
      }
      if ((sscanf(str, "%f", &val) != 1)) {
        printf("invalid file format");
        exit(-1);
      }
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);	
}


/* compute N time steps */
int compute_tran_temp(
    sycl::queue &q,
    float *MatrixPower, 
    float *MatrixTemp[2], 
    int col, int row,
    int total_iterations, int num_iterations, 
    int blockCols, int blockRows, int borderCols, int borderRows)
{ 
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  int t;
#ifdef DEBUG
  printf("%f %f %f %f %f %f %f\n", grid_height,grid_width,Cap,Rx,Ry,Rz,step);
#endif

  int src = 0, dst = 1;

  // Determine GPU work group grid
  size_t global_work_size[2];
  global_work_size[0] = BLOCK_SIZE * blockCols;
  global_work_size[1] = BLOCK_SIZE * blockRows;

  size_t local_work_size[2];
  local_work_size[0] = BLOCK_SIZE;
  local_work_size[1] = BLOCK_SIZE;

  sycl::range<2> gws (global_work_size[1], global_work_size[0]);
  sycl::range<2> lws (local_work_size[1], local_work_size[0]);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (t = 0; t < total_iterations; t += num_iterations) {

    // Specify kernel arguments
    int iter = MIN(num_iterations, total_iterations - t);

    q.submit([&](sycl::handler& cgh) {
      auto power_acc = MatrixPower;
      auto temp_src_acc = MatrixTemp[src];
      auto temp_dst_acc = MatrixTemp[dst];
      sycl::local_accessor <float, 2> temp_on_device (sycl::range<2>{BLOCK_SIZE,BLOCK_SIZE}, cgh);
      sycl::local_accessor <float, 2> power_on_device (sycl::range<2>{BLOCK_SIZE,BLOCK_SIZE}, cgh);
      sycl::local_accessor <float, 2> temp_t (sycl::range<2>{BLOCK_SIZE,BLOCK_SIZE}, cgh);
      cgh.parallel_for<class calc_temp>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
         #include "kernel_hotspot.sycl"
      });
    });

    // Swap input and output GPU matrices
    src = 1 - src;
    dst = 1 - dst;
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time %f (s)\n", time * 1e-9f);

  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char** argv) {

  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  int size, size_bytes;
  int grid_rows,grid_cols = 0;
  float *FilesavingTemp,*FilesavingPower;
  char *tfile, *pfile, *ofile;

  int total_iterations = 60;  // this can be overwritten by the commandline argument
  int pyramid_height = 1;     // step size

  if (argc < 7) usage(argc, argv);

  if((grid_rows = atoi(argv[1]))<=0||
     (grid_cols = atoi(argv[1]))<=0||
     (pyramid_height = atoi(argv[2]))<=0||
     (total_iterations = atoi(argv[3]))<=0)
    usage(argc, argv);

  tfile=argv[4];
  pfile=argv[5];
  ofile=argv[6];

  size = grid_rows*grid_cols;
  size_bytes = size * sizeof(float);

  // --------------- pyramid parameters --------------- 
  int borderCols = (pyramid_height)*EXPAND_RATE/2;
  int borderRows = (pyramid_height)*EXPAND_RATE/2;
  int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
  int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
  int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

  FilesavingTemp = (float *) malloc(size_bytes);
  FilesavingPower = (float *) malloc(size_bytes);

  if( !FilesavingPower || !FilesavingTemp) {
    printf("unable to allocate memory");
    exit(-1);
  }

  // Read input data from disk
  readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
  readinput(FilesavingPower, grid_rows, grid_cols, pfile);

  long long start_time = get_time();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *MatrixPower = sycl::malloc_device<float>(size, q);
  q.memcpy(MatrixPower, FilesavingPower, size_bytes);

  float *MatrixTemp[2];
  MatrixTemp[0] = sycl::malloc_device<float>(size, q);
  MatrixTemp[1] = sycl::malloc_device<float>(size, q);
  q.memcpy(MatrixTemp[0], FilesavingTemp, size_bytes);

  // Perform the computation
  int ret = compute_tran_temp(q, MatrixPower, MatrixTemp, grid_cols, grid_rows, 
      total_iterations, pyramid_height, blockCols, blockRows, borderCols, borderRows);

  q.memcpy(FilesavingPower, MatrixTemp[ret], size_bytes).wait();

  long long end_time = get_time();
  printf("Device offloading time: %.3f seconds\n", ((float) (end_time - start_time)) / (1000*1000));

  // Write final output to output file
  writeoutput(FilesavingPower, grid_rows, grid_cols, ofile);

  sycl::free(MatrixPower, q);
  sycl::free(MatrixTemp[0], q);
  sycl::free(MatrixTemp[1], q);
  free(FilesavingTemp);
  free(FilesavingPower);

  return 0;
}
