#include <sys/types.h>
#include <chrono>
#include <sycl/sycl.hpp>
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

void hotspot3d(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const float*__restrict__ tIn,
    const float*__restrict__ pIn,
          float*__restrict__ tOut,
    const int numCols,
    const int numRows,
    const int layers,
    const float ce,
    const float cw,
    const float cn,
    const float cs,
    const float ct,
    const float cb,
    const float cc,
    const float stepDivCap)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      float amb_temp = 80.0;

      int i = item.get_global_id(2);
      int j = item.get_global_id(1);
      int c = i + j * numCols;
      int xy = numCols * numRows;

      int W = (i == 0)        ? c : c - 1;
      int E = (i == numCols-1)? c : c + 1;
      int N = (j == 0)        ? c : c - numCols;
      int S = (j == numRows-1)? c : c + numCols;

      float temp1, temp2, temp3;
      temp1 = temp2 = tIn[c];
      temp3 = tIn[c+xy];
      tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
      c += xy;
      W += xy;
      E += xy;
      N += xy;
      S += xy;

      for (int k = 1; k < layers-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
          + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
      }
      temp1 = temp2;
      temp2 = temp3;
      tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S]
        + cn * tIn[N] + cb * temp1 + ct * temp3 + stepDivCap * pIn[c] + ct * amb_temp;
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
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
  ce = cw = stepDivCap/ Rx;
  cn = cs = stepDivCap/ Ry;
  ct = cb = stepDivCap/ Rz;
  cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  int size = numCols * numRows * layers;
  size_t size_byte = size * sizeof(float);
  float* tIn   = (float*) calloc(size, sizeof(float));
  float* pIn   = (float*) calloc(size, sizeof(float));
  float* tCopy = (float*) malloc(size_byte);
  float* tOut  = (float*) calloc(size, sizeof(float));

  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global_work_size[2];
  size_t local_work_size[2];
  memcpy(tCopy, tIn, size_byte);

  long long start = get_time();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_tIn = sycl::malloc_device<float>(size, q);
  float *d_pIn = sycl::malloc_device<float>(size, q);
  float *d_tOut = sycl::malloc_device<float>(size, q);

  q.memcpy(d_tIn, tIn, size_byte);
  q.memcpy(d_pIn, pIn, size_byte);

  global_work_size[1] = numCols;
  global_work_size[0] = numRows;

  local_work_size[1] = WG_SIZE_X;
  local_work_size[0] = WG_SIZE_Y;

  sycl::range<3> gws (1, global_work_size[0], global_work_size[1]);
  sycl::range<3> lws (1, local_work_size[0], local_work_size[1]);

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  for(int j = 0; j < iterations; j++)
  {
    hotspot3d(q, gws, lws, 0, d_tIn, d_pIn, d_tOut, numCols, numRows, layers,
              ce, cw, cn, cs, ct, cb, cc, stepDivCap);

    float* temp = d_tIn;
    d_tIn = d_tOut;
    d_tOut = temp;
  }

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Average kernel execution time %f (us)\n", (ktime * 1e-3f) / iterations);

  float* d_sel = (iterations & 01) ? d_tIn : d_tOut;
  q.memcpy(tOut, d_sel, size_byte).wait();

  sycl::free(d_tIn, q);
  sycl::free(d_pIn, q);
  sycl::free(d_tOut, q);

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
