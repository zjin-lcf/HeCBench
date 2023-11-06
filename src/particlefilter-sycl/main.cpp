#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <sycl/sycl.hpp>

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

/**
  @var M value for Linear Congruential Generator (LCG); use GCC's value
  */
//long M = INT_MAX;
/**
  @var A value for LCG
  */
//int A = 1103515245;
/**
  @var C value for LCG
  */
//int C = 12345;

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/*
   @brief cleans up the OpenCL framework
   */
/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/
long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) +tv.tv_usec;
}

/* Returns the number of seconds elapsed between the two specified times */
float elapsed_time(long long start_time, long long end_time) {
  return (float) (end_time - start_time) / (1000 * 1000);
}

/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */
float randu(int * seed, int index) {
  int num = A * seed[index] + C;
  seed[index] = num % M;
  return std::fabs(seed[index] / ((float) M));
}

/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a float representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */
float randn(int * seed, int index) {
  /*Box-Muller algorithm*/
  float u = randu(seed, index);
  float v = randu(seed, index);
  float cosine = std::cos(2 * PI * v);
  float rt = -2 * std::log(u);
  return std::sqrt(rt) * cosine;
}

/**
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
float roundFloat(float value) {
  int newValue = (int) (value);
  if (value - newValue < .5)
    return newValue;
  else
    return newValue++;
}

/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */
void setIf(int testValue, int newValue, unsigned char * array3D, int * dimX, int * dimY, int * dimZ) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        if (array3D[x * *dimY * *dimZ + y * *dimZ + z] == testValue)
          array3D[x * *dimY * *dimZ + y * *dimZ + z] = newValue;
      }
    }
  }
}

/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */
void addNoise(unsigned char * array3D, int * dimX, int * dimY, int * dimZ, int * seed) {
  int x, y, z;
  for (x = 0; x < *dimX; x++) {
    for (y = 0; y < *dimY; y++) {
      for (z = 0; z < *dimZ; z++) {
        array3D[x * *dimY * *dimZ + y * *dimZ + z] = array3D[x * *dimY * *dimZ + y * *dimZ + z] + (unsigned char) (5 * randn(seed, 0));
      }
    }
  }
}

/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */
void strelDisk(int * disk, int radius) {
  int diameter = radius * 2 - 1;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      float distance = std::sqrt(pow((float) (x - radius + 1), 2) + pow((float) (y - radius + 1), 2));
      if (distance < radius)
        disk[x * diameter + y] = 1;
      else
        disk[x * diameter + y] = 0;
    }
  }
}

/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */
void dilate_matrix(unsigned char * matrix, int posX, int posY, int posZ, int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0)
    startX++;
  int startY = posY - error;
  while (startY < 0)
    startY++;
  int endX = posX + error;
  while (endX > dimX)
    endX--;
  int endY = posY + error;
  while (endY > dimY)
    endY--;
  int x, y;
  for (x = startX; x < endX; x++) {
    for (y = startY; y < endY; y++) {
      float distance = std::sqrt(std::pow((float) (x - posX), 2) + std::pow((float) (y - posY), 2));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}

/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */
void imdilate_disk(unsigned char * matrix, int dimX, int dimY, int dimZ, int error, unsigned char * newMatrix) {
  int x, y, z;
  for (z = 0; z < dimZ; z++) {
    for (x = 0; x < dimX; x++) {
      for (y = 0; y < dimY; y++) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
        }
      }
    }
  }
}

/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */
void getneighbors(int * se, int numOnes, int * neighbors, int radius) {
  int x, y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int) (y - center);
        neighbors[neighY * 2 + 1] = (int) (x - center);
        neighY++;
      }
    }
  }
}

/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */
void videoSequence(unsigned char * I, int IszX, int IszY, int Nfr, int * seed) {
  int k;
  int max_size = IszX * IszY * Nfr;
  /*get object centers*/
  int x0 = (int) roundFloat(IszY / 2.0);
  int y0 = (int) roundFloat(IszX / 2.0);
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  /*move point*/
  int xk, yk, pos;
  for (k = 1; k < Nfr; k++) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  /*dilate matrix*/
  unsigned char * newMatrix = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  int x, y;
  for (x = 0; x < IszX; x++) {
    for (y = 0; y < IszY; y++) {
      for (k = 0; k < Nfr; k++) {
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);

  /*define background, add noise*/
  setIf(0, 100, I, &IszX, &IszY, &Nfr);
  setIf(1, 228, I, &IszX, &IszY, &Nfr);
  /*add noise*/
  addNoise(I, &IszX, &IszY, &Nfr, seed);

}

/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */
int findIndex(float * CDF, int lengthCDF, float value) {
  int index = -1;
  int x;
  for (x = 0; x < lengthCDF; x++) {
    if (CDF[x] >= value) {
      index = x;
      break;
    }
  }
  if (index == -1) {
    return lengthCDF - 1;
  }
  return index;
}

/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */
int particleFilter(unsigned char * I, int IszX, int IszY, int Nfr, int * seed, int Nparticles) {
  int max_size = IszX * IszY*Nfr;
  //original particle centroid
  float xe = roundFloat(IszY / 2.0);
  float ye = roundFloat(IszX / 2.0);

  //expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int * disk = (int*) calloc(diameter * diameter, sizeof (int));
  strelDisk(disk, radius);
  int countOnes = 0;
  int x, y;
  for (x = 0; x < diameter; x++) {
    for (y = 0; y < diameter; y++) {
      if (disk[x * diameter + y] == 1)
        countOnes++;
    }
  }
  int * objxy = (int *) calloc(countOnes * 2, sizeof(int));
  getneighbors(disk, countOnes, objxy, radius);

  //initial weights are all equal (1/Nparticles)
  float * weights = (float *) calloc(Nparticles, sizeof(float));
  for (x = 0; x < Nparticles; x++) {
    weights[x] = 1 / ((float) (Nparticles));
  }
  /****************************************************************
   **************   B E G I N   A L L O C A T E *******************
   ****************************************************************/
  float * likelihood = (float *) calloc(Nparticles + 1, sizeof (float));
  float * arrayX = (float *) calloc(Nparticles, sizeof (float));
  float * arrayY = (float *) calloc(Nparticles, sizeof (float));
  float * xj = (float *) calloc(Nparticles, sizeof (float));
  float * yj = (float *) calloc(Nparticles, sizeof (float));
  float * CDF = (float *) calloc(Nparticles, sizeof(float));

  //GPU copies of arrays
  int * ind = (int*) calloc(countOnes * Nparticles, sizeof(int));
  float * u = (float *) calloc(Nparticles, sizeof(float));

  //Donnie - this loop is different because in this kernel, arrayX and arrayY
  //  are set equal to xj before every iteration, so effectively, arrayX and
  //  arrayY will be set to xe and ye before the first iteration.
  for (x = 0; x < Nparticles; x++) {
    xj[x] = xe;
    yj[x] = ye;
  }

  long long offload_start = get_time();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int k;
  const int threads_per_block = BLOCK_SIZE;

  int num_blocks = std::ceil((float) Nparticles / (float) threads_per_block);
  sycl::range<1> lws (threads_per_block);
  sycl::range<1> gws (num_blocks*threads_per_block);
#ifdef DEBUG
  printf("BLOCK_SIZE=%d \n",BLOCK_SIZE);
#endif

  float *likelihood_d = sycl::malloc_device<float>(Nparticles + 1, q);
  float *arrayX_d = sycl::malloc_device<float>(Nparticles, q);
  float *arrayY_d = sycl::malloc_device<float>(Nparticles, q);

  float *xj_d = sycl::malloc_device<float>(Nparticles, q);
  float *yj_d = sycl::malloc_device<float>(Nparticles, q);
  q.memcpy(xj_d, xj, Nparticles*sizeof(float));
  q.memcpy(yj_d, yj, Nparticles*sizeof(float));

  float *CDF_d = sycl::malloc_device<float>(Nparticles, q);
  float *u_d = sycl::malloc_device<float>(Nparticles, q);
  int *ind_d = sycl::malloc_device<int>(countOnes * Nparticles, q);

  float *weights_d = sycl::malloc_device<float>(Nparticles, q);
  q.memcpy(weights_d, weights, Nparticles*sizeof(float));

  unsigned char *I_d = sycl::malloc_device<unsigned char>(IszX * IszY * Nfr, q);
  q.memcpy(I_d, I, IszX * IszY * Nfr * sizeof(unsigned char));

  int *seed_d = sycl::malloc_device<int>(Nparticles, q);
  q.memcpy(seed_d, seed, Nparticles*sizeof(int));

  float *partial_sums_d = sycl::malloc_device<float>(Nparticles+1, q);

  int *objxy_d = sycl::malloc_device<int>(2*countOnes, q);
  q.memcpy(objxy_d, objxy, 2*countOnes*sizeof(int));

  q.wait();
  long long start = get_time();

  for (k = 1; k < Nfr; k++) {
    /****************** L I K E L I H O O D ************************************/
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> weights_local(sycl::range<1>(threads_per_block), cgh);
      cgh.parallel_for<class likelihood>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        #include "kernel_likelihood.sycl"
      });
    });

#ifdef DEBUG
    float * sum = (float *) calloc(Nparticles + 1, sizeof (float));
    q.memcpy(sum, partial_sums_d, (Nparticles+1)*sizeof(float)).wait();
    for (int i = 0; i < Nparticles+1; i++)
      printf("%f ", sum[i]);
    printf("\n");
#endif

    q.submit([&](sycl::handler& cgh) {
      cgh.single_task<class sum>([=]() {
        #include "kernel_sum.sycl"
      });
    });

#ifdef DEBUG
    // this shows the sum of all partial_sum results
    q.memcpy(sum, partial_sums_d, sizeof(float)).wait();
    printf("kernel sum: frame=%d partial_sums[0]=%f\n", k, sum[0]);
    free(sum);
#endif

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 0> u1(cgh);
      sycl::local_accessor<float, 0> sumWeights(cgh);
      cgh.parallel_for<class normalize_weights>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        #include "kernel_normalize_weights.sycl"
      });
    });

    //Set number of threads
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class find_index>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        #include "kernel_find_index.sycl"
      });
    });
  } //end loop

  q.wait();
  long long end = get_time();
  printf("Average execution time of kernels: %f (s)\n",
         elapsed_time(start, end) / (Nfr-1));

  q.memcpy(arrayX, arrayX_d, Nparticles*sizeof(float));
  q.memcpy(arrayY, arrayY_d, Nparticles*sizeof(float));
  q.memcpy(weights, weights_d, Nparticles*sizeof(float));
  q.wait();

  sycl::free(likelihood_d, q);
  sycl::free(arrayX_d, q);
  sycl::free(arrayY_d, q);
  sycl::free(xj_d, q);
  sycl::free(yj_d, q);
  sycl::free(CDF_d, q);
  sycl::free(partial_sums_d, q);
  sycl::free(objxy_d, q);
  sycl::free(u_d, q);
  sycl::free(ind_d, q);
  sycl::free(seed_d, q);
  sycl::free(weights_d, q);
  sycl::free(I_d, q);

  long long offload_end = get_time();
  printf("Device offloading time: %lf (s)\n", elapsed_time(offload_start, offload_end));

  xe = 0;
  ye = 0;
  // estimate the object location by expected values
  for (x = 0; x < Nparticles; x++) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = std::sqrt(std::pow((float) (xe - (int) roundFloat(IszY / 2.0)), 2) +
      std::pow((float) (ye - (int) roundFloat(IszX / 2.0)), 2));

  //Output results
  FILE *fid;
  fid=fopen("output.txt", "w+");
  if( fid == NULL ){
    printf( "The file was not opened for writing\n" );
    return -1;
  }
  fprintf(fid, "XE: %lf\n", xe);
  fprintf(fid, "YE: %lf\n", ye);
  fprintf(fid, "distance: %lf\n", distance);
  fclose(fid);

  //free regular memory
  free(likelihood);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
  return 0;
}

int main(int argc, char * argv[]) {

  const char* usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
  //check number of arguments
  if (argc != 9) {
    printf("%s\n", usage);
    return 0;
  }
  //check args deliminators
  if (strcmp(argv[1], "-x") || strcmp(argv[3], "-y") || strcmp(argv[5], "-z") || strcmp(argv[7], "-np")) {
    printf("%s\n", usage);
    return 0;
  }

  int IszX, IszY, Nfr, Nparticles;

  //converting a string to a integer
  if (sscanf(argv[2], "%d", &IszX) == EOF) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }

  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[4], "%d", &IszY) == EOF) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }

  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[6], "%d", &Nfr) == EOF) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }

  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }

  //converting a string to a integer
  if (sscanf(argv[8], "%d", &Nparticles) == EOF) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }

  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }

#ifdef DEBUG
  printf("dimX=%d dimY=%d Nfr=%d Nparticles=%d\n",
      IszX, IszY, Nfr, Nparticles);
#endif

  //establish seed
  int * seed = (int *) calloc(Nparticles, sizeof(int));
  int i;
  for (i = 0; i < Nparticles; i++)
    seed[i] = i+1;

  //calloc matrix
  unsigned char * I = (unsigned char *) calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  long long start = get_time();

  //call video sequence
  videoSequence(I, IszX, IszY, Nfr, seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f\n", elapsed_time(start, endVideoSequence));

  //call particle filter
  particleFilter(I, IszX, IszY, Nfr, seed, Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f\n", elapsed_time(endVideoSequence, endParticleFilter));

  printf("ENTIRE PROGRAM TOOK %f\n", elapsed_time(start, endParticleFilter));

  free(seed);
  free(I);
  return 0;
}
