#include <sys/types.h>
#include <chrono>
#include "common.h"
#include "3D_helper.h"

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows/cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}

int main(int argc, char** argv)
{
  if (argc != 7)
  {
    usage(argc,argv);
  }

  char *pfile, *tfile, *ofile;
  int iterations = atoi(argv[3]);

  pfile            = argv[4];
  tfile            = argv[5];
  ofile            = argv[6];
  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[1]);
  int layers       = atoi(argv[2]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  float* tIn   = (float*) calloc(size,sizeof(float));
  float* pIn   = (float*) calloc(size,sizeof(float));
  float* tCopy = (float*) malloc(size * sizeof(float));
  float* tOut  = (float*) calloc(size,sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global_work_size[2];                   
  size_t local_work_size[2];
  memcpy(tCopy,tIn, size * sizeof(float));

  long long start = get_time();

  { // SYCL scope
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();
    buffer<float, 1> d_tIn (tIn, size, props);
    buffer<float, 1> d_pIn (pIn, size, props);
    buffer<float, 1> d_tOut (tOut, size, props);
    d_tIn.set_final_data(nullptr);
    d_tOut.set_final_data(nullptr);

    global_work_size[1] = numCols;
    global_work_size[0] = numRows;

    local_work_size[1] = WG_SIZE_X;
    local_work_size[0] = WG_SIZE_Y;

    range<2> gws (global_work_size[0], global_work_size[1]);
    range<2> lws (local_work_size[0], local_work_size[1]);

    q.wait();
    auto kstart = std::chrono::steady_clock::now();

    for(int j = 0; j < iterations; j++)
    {
      q.submit([&](handler& cgh) {
        auto pIn_acc = d_pIn.get_access<sycl_read>(cgh); 
        auto tIn_acc = d_tIn.get_access<sycl_read>(cgh);
        auto tOut_acc = d_tOut.get_access<sycl_discard_write>(cgh);

        cgh.parallel_for<class hotspot>(
          nd_range<2>(gws, lws), [=] (nd_item<2> item) {
            #include "kernel_hotspot.sycl"
        });
      });

      auto temp = std::move(d_tIn);
      d_tIn = std::move(d_tOut);
      d_tOut = std::move(temp);
    }

    q.wait();
    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

    q.submit([&](handler& cgh) {
      auto d_sel = (iterations & 01) ? d_tIn : d_tOut;
      auto d_tOut_acc = d_sel.get_access<sycl_read>(cgh);
      cgh.copy(d_tOut_acc, tOut);
    }).wait();

  } // SYCL scope

  long long stop = get_time();

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);

  float acc = accuracy(tOut,answer,numRows*numCols*layers);
  float time = (float)((stop - start)/(1000.0 * 1000.0));
  printf("Device offloading time: %.3f (s)\n",time);
  printf("Root-mean-square error: %e\n",acc);

  writeoutput(tOut,numRows,numCols,layers,ofile);

  free(answer);
  free(tIn);
  free(pIn);
  free(tCopy);
  free(tOut);
  return 0;
}
