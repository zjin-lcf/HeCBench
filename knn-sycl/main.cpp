/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include <cstdio>
#include <sys/time.h>
#include <time.h>
#include "common.h"

// Constants used by the program
#define BLOCK_DIM 16

//-----------------------------------------------------------------------------------------------//
//                                   K-th NEAREST NEIGHBORS //
//-----------------------------------------------------------------------------------------------//

/*
  * @param ref_host      reference points ; pointer to linear matrix
  * @param ref_width     number of reference points ; width of the matrix
  * @param query_host    query points ; pointer to linear matrix
  * @param query_width   number of query points ; width of the matrix
  * @param height        dimension of points ; height of the matrices
  * @param k             number of neighbor to consider
  * @param dist_host     distances to k nearest neighbors ; pointer to linear
 * matrix
  * @param dist_host     indexes of the k nearest neighbors ; pointer to linear
 * matrix
  *
  */
void knn_parallel(queue &q, float *ref_host, int ref_width, float *query_host,
              int query_width, int height, int k, float *dist_host, int *ind_host) {

  // Allocation of global memory for indexes CUDA_CHECK
  buffer<float, 1> ref_dev (ref_host, ref_width * height);
  buffer<float, 1> query_dev (query_host, query_width * height);
  buffer<float, 1> dist_dev (query_width * ref_width);
  buffer<int, 1> ind_dev (query_width * k);

  // Grids ans threads
  range<2> g_16x16((ref_width + 15) / 16 * 16, (query_width + 15) / 16 * 16);
  range<2> t_16x16(16, 16);

  range<1> g_256x1((query_width + 255) / 256 * 256);
  range<1> t_256x1(256);

  range<2> g_k_16x16((k + 15) / 16 * 16, (query_width + 15) / 16 * 16);
  range<2> t_k_16x16(16, 16);


  // Kernel 1: Compute all the distances
  //cuComputeDistanceGlobal<<<g_16x16, t_16x16>>>(ref_dev, ref_width, query_dev, query_width, height, dist_dev);
  q.submit([&] (handler &h) {
    auto A = ref_dev.get_access<sycl_read>(h);
    auto B = query_dev.get_access<sycl_read>(h);
    auto AB = dist_dev.get_access<sycl_write>(h);
    accessor<float, 2, sycl_read_write, access::target::local> shared_A ({BLOCK_DIM, BLOCK_DIM}, h);
    accessor<float, 2, sycl_read_write, access::target::local> shared_B ({BLOCK_DIM, BLOCK_DIM}, h);
    accessor<int, 1, sycl_read_write, access::target::local> begin_A (1, h);
    accessor<int, 1, sycl_read_write, access::target::local> begin_B (1, h);
    accessor<int, 1, sycl_read_write, access::target::local> step_A (1, h);
    accessor<int, 1, sycl_read_write, access::target::local> step_B (1, h);
    accessor<int, 1, sycl_read_write, access::target::local> end_A (1, h);
    h.parallel_for(nd_range<2>(g_16x16, t_16x16), [=] (nd_item<2> item) {
      // Thread index
      int tx = item.get_local_id(1);
      int ty = item.get_local_id(0);

      // Other variables
      float tmp;
      float ssd = 0;

      // Loop parameters
      begin_A[0] = BLOCK_DIM * item.get_group(0);
      begin_B[0] = BLOCK_DIM * item.get_group(1);
      step_A[0]  = BLOCK_DIM * ref_width;
      step_B[0]  = BLOCK_DIM * query_width;
      end_A[0]   = begin_A[0] + (height - 1) * ref_width;

      // Conditions
      int cond0 = (begin_A[0] + tx < ref_width); // used to write in shared memory
      int cond1 = (begin_B[0] + tx < query_width); // used to write in shared memory & to
                                       // computations and to write in output matrix
      int cond2 =
          (begin_A[0] + ty < ref_width); // used to computations and to write in output matrix

      // Loop over all the sub-matrices of A and B required to compute the block
      // sub-matrix
      for (int a = begin_A[0], b = begin_B[0]; 
               a <= end_A[0]; a += step_A[0], b += step_B[0]) {
        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        if (a / ref_width + ty < height) {
          shared_A[ty][tx] = (cond0) ? A[a + ref_width * ty + tx] : 0;
          shared_B[ty][tx] = (cond1) ? B[b + query_width * ty + tx] : 0;
        } else {
          shared_A[ty][tx] = 0;
          shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        item.barrier(access::fence_space::local_space);

        // Compute the difference between the two matrixes; each thread computes one
        // element of the block sub-matrix
        if (cond2 && cond1) {
          for (int k = 0; k < BLOCK_DIM; ++k) {
            tmp = shared_A[k][ty] - shared_B[k][tx];
            ssd += tmp * tmp;
          }
        }

        // Synchronize to make sure that the preceding computation is done before
        // loading two new sub-matrices of A and B in the next iteration
        item.barrier(access::fence_space::local_space);
      }

      // Write the block sub-matrix to device memory; each thread writes one element
      if (cond2 && cond1)
        AB[(begin_A[0] + ty) * query_width + begin_B[0] + tx] = ssd;
    });
  });

#ifdef DEBUG
  q.submit([&] (handler &h) {
    auto AB_h = dist_dev.get_access<sycl_read>(h);
    h.copy(AB_h, dist_host);
    });
  q.wait();
  for (int i = 0; i < query_width * ref_width; i++)
    printf("k1 dist: %d %f\n", i, dist_host[i]);
#endif

  // Kernel 2: Sort each column
  //cuInsertionSort<<<g_256x1, t_256x1>>>(dist_dev, ind_dev, query_width, ref_width, k);
  q.submit([&] (handler &h) {
    auto dist = dist_dev.get_access<sycl_read_write>(h);
    auto ind = ind_dev.get_access<sycl_read_write>(h);
    h.parallel_for(nd_range<1>(g_256x1, t_256x1), [=] (nd_item<1> item) {
      // Variables
      int l, i, j;
      float *p_dist;
      int *p_ind;
      float curr_dist, max_dist;
      int curr_row, max_row;
      unsigned int xIndex = item.get_global_id(0);

      if (xIndex < query_width) {
        // Pointer shift, initialization, and max value
        //p_dist = &dist[xIndex];
        //p_ind = &ind[xIndex];
        //max_dist = p_dist[0];
        //p_ind[0] = 1;
        max_dist = dist[xIndex];
        ind[xIndex] = 0;

        // Part 1 : sort kth firt elementZ
        for (l = 1; l < k; l++) {
          curr_row = l * query_width;
          //curr_dist = p_dist[curr_row];
          curr_dist = dist[xIndex+curr_row];
          if (curr_dist < max_dist) {
            i = l - 1;
            for (int a = 0; a < l - 1; a++) {
              //if (p_dist[a * query_width] > curr_dist) {
              if (dist[xIndex + a * query_width] > curr_dist) {
                i = a;
                break;
              }
            }
            for (j = l; j > i; j--) {
              dist[xIndex+j * query_width] = dist[xIndex+(j - 1) * query_width];
              ind[xIndex + j * query_width] = ind[xIndex + (j - 1) * query_width];
              //p_dist[j * query_width] = p_dist[(j - 1) * query_width];
              //p_ind[j * query_width] = p_ind[(j - 1) * query_width];
            }
            //p_dist[i * query_width] = curr_dist;
              dist[xIndex+i * query_width] = curr_dist;
            //p_ind[i * query_width] = l + 1;
              ind[xIndex+i * query_width] = l;
          } else {
            //p_ind[l * query_width] = l + 1;
              ind[xIndex+l * query_width] = l;
          }
          //max_dist = p_dist[curr_row];
          max_dist = dist[xIndex + curr_row];
        }

        // Part 2 : insert element in the k-th first lines
        max_row = (k - 1) * query_width;
        for (l = k; l < ref_width; l++) {
          //curr_dist = p_dist[l * query_width];
          curr_dist = dist[xIndex + l * query_width];
          if (curr_dist < max_dist) {
            i = k - 1;
            for (int a = 0; a < k - 1; a++) {
              //if (p_dist[a * query_width] > curr_dist) {
              if (dist[xIndex + a * query_width] > curr_dist) {
                i = a;
                break;
              }
            }
            for (j = k - 1; j > i; j--) {
              //p_dist[j * query_width] = p_dist[(j - 1) * query_width];
              //p_ind[j * query_width] = p_ind[(j - 1) * query_width];
              dist[xIndex+j * query_width] = dist[xIndex+(j - 1) * query_width];
              ind[xIndex + j * query_width] = ind[xIndex + (j - 1) * query_width];
            }
            //p_dist[i * query_width] = curr_dist;
            //p_ind[i * query_width] = l + 1;
            //max_dist = p_dist[max_row];
            dist[xIndex+i * query_width] = curr_dist;
            ind[xIndex+i * query_width] = l;
            max_dist = dist[xIndex + max_row];
          }
        }
      }

      /*
      if (xIndex < query_width) {
        // Pointer shift, initialization, and max value
        p_dist = &dist[xIndex];
        p_ind = &ind[xIndex];
        max_dist = p_dist[0];
        p_ind[0] = 1;

        // Part 1 : sort kth firt elementZ
        for (l = 1; l < k; l++) {
          curr_row = l * query_width;
          curr_dist = p_dist[curr_row];
          if (curr_dist < max_dist) {
            i = l - 1;
            for (int a = 0; a < l - 1; a++) {
              if (p_dist[a * query_width] > curr_dist) {
                i = a;
                break;
              }
            }
            for (j = l; j > i; j--) {
              p_dist[j * query_width] = p_dist[(j - 1) * query_width];
              p_ind[j * query_width] = p_ind[(j - 1) * query_width];
            }
            p_dist[i * query_width] = curr_dist;
            p_ind[i * query_width] = l + 1;
          } else {
            p_ind[l * query_width] = l + 1;
          }
          max_dist = p_dist[curr_row];
        }

        // Part 2 : insert element in the k-th first lines
        max_row = (k - 1) * query_width;
        for (l = k; l < ref_width; l++) {
          curr_dist = p_dist[l * query_width];
          if (curr_dist < max_dist) {
            i = k - 1;
            for (int a = 0; a < k - 1; a++) {
              if (p_dist[a * query_width] > curr_dist) {
                i = a;
                break;
              }
            }
            for (j = k - 1; j > i; j--) {
              p_dist[j * query_width] = p_dist[(j - 1) * query_width];
              p_ind[j * query_width] = p_ind[(j - 1) * query_width];
            }
            p_dist[i * query_width] = curr_dist;
            p_ind[i * query_width] = l + 1;
            max_dist = p_dist[max_row];
          }
        }
      }
      */
    });
  });

#ifdef DEBUG
  q.submit([&] (handler &h) {
    auto AB_h = dist_dev.get_access<sycl_read>(h);
    h.copy(AB_h, dist_host);
    });
  q.wait();
  for (int i = 0; i < query_width * ref_width; i++)
    printf("k2 dist: %d %f\n", i, dist_host[i]);

  q.submit([&] (handler &h) {
    auto AB_h = ind_dev.get_access<sycl_read>(h);
    h.copy(AB_h, ind_host);
    });
  q.wait();
  for (int i = 0; i < query_width * k; i++)
    printf("k2 index: %d %d\n", i, ind_host[i]);
#endif

  // Kernel 3: Compute square root of k first elements
  //cuParallelSqrt<<<g_k_16x16, t_k_16x16>>>(dist_dev, query_width, k);
  q.submit([&] (handler &h) {
    auto dist = dist_dev.get_access<sycl_write>(h);
    h.parallel_for(nd_range<2>(g_k_16x16, t_k_16x16), [=] (nd_item<2> item) {
      unsigned int xIndex = item.get_global_id(1);
      unsigned int yIndex = item.get_global_id(0);
      if (xIndex < query_width && yIndex < k)
        dist[yIndex * query_width + xIndex] = sqrt(dist[yIndex * query_width + xIndex]);
    });
  });

  q.submit([&] (handler &h) {
    auto dist_dev_acc = dist_dev.get_access<sycl_read>(h, range<1>(query_width * k));
    h.copy(dist_dev_acc, dist_host);
  });

  q.submit([&] (handler &h) {
    auto ind_dev_acc = ind_dev.get_access<sycl_read>(h);
    h.copy(ind_dev_acc, ind_host);
  });
  q.wait();

}

float compute_distance(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int ref_index, int query_index) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float diff =
        ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

void modified_insertion_sort(float *dist, int *index, int length, int k) {

  // Initialise the first index
  index[0] = 0;

  // Go through all points
  for (int i = 1; i < length; ++i) {

    // Store current distance and associated index
    float curr_dist = dist[i];
    int curr_index = i;

    // Skip the current value if its index is >= k and if it's higher the k-th
    // slready sorted mallest value
    if (i >= k && curr_dist >= dist[k - 1]) {
      continue;
    }

    // Shift values (and indexes) higher that the current distance to the right
    int j = min(i, k - 1);
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j] = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }

    // Write the current distance and index at their position
    dist[j] = curr_dist;
    index[j] = curr_index;
  }
}

bool knn_serial(const float *ref, int ref_nb, const float *query, int query_nb,
           int dim, int k, float *knn_dist, int *knn_index) {
  // Allocate local array to store all the distances / indexes for a given query
  // point
  float *dist = (float *)malloc(ref_nb * sizeof(float));
  int *index = (int *)malloc(ref_nb * sizeof(int));

  // Allocation checks
  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist);
    free(index);
    return false;
  }

  // Process one query point at the time
  for (int i = 0; i < query_nb; ++i) {

    // Compute all distances / indexes
    for (int j = 0; j < ref_nb; ++j) {
      dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }

    // Sort distances / indexes
    modified_insertion_sort(dist, index, ref_nb, k);

    // Copy k smallest distances and their associated index
    for (int j = 0; j < k; ++j) {
      knn_dist[j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }

  // Memory clean-up
  free(dist);
  free(index);
  return true;
}

int main(void) {
  float *ref;          // Pointer to reference point array
  float *query;        // Pointer to query point array
  float *dist;         // Pointer to distance array
  int *ind;            // Pointer to index array
  int ref_nb = 256;   // Reference point number, max=65535
  int query_nb = 256; // Query point number,     max=65535
  int dim = 32;        // Dimension of points
  int k = 20;          // Nearest neighbors to consider
  int iterations = 1;
  int c_iterations = 1;
  int i;
  const float precision = 0.001f; // distance error max
  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  // Memory allocation
  ref = (float *)malloc(ref_nb * dim * sizeof(float));
  query = (float *)malloc(query_nb * dim * sizeof(float));
  dist = (float *)malloc(query_nb * k * sizeof(float));
  ind = (int *)malloc(query_nb * k * sizeof(float));

  // Init
  srand(2);
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;


  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  int *knn_index = (int *)malloc(query_nb * k * sizeof(int));
  printf("Ground truth computation in progress...\n\n");
  if (!knn_serial(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
    return EXIT_FAILURE;
  }

  struct timeval tic;
  struct timeval toc;
  float elapsed_time;

  printf("On CPU: \n");
  gettimeofday(&tic, NULL);
  for (i = 0; i < c_iterations; i++) {
    knn_serial(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         c_iterations, elapsed_time / (c_iterations));


#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  printf("on GPU: \n");
  gettimeofday(&tic, NULL);
  for (i = 0; i < iterations; i++) {
    knn_parallel(q, ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  gettimeofday(&toc, NULL);
  elapsed_time = toc.tv_sec - tic.tv_sec;
  elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
  printf(" done in %f s for %d iterations (%f s by iteration)\n", elapsed_time,
         iterations, elapsed_time / (iterations));

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind[i] == knn_index[i]) {
      nb_correct_indexes++;
    }
  }

  float precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  float index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("Precision accuracy %f\nIndex accuracy %f\n", precision_accuracy, index_accuracy);

  free(ind);
  free(dist);
  free(query);
  free(ref);
}
