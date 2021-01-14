/*
 * CUDA Kernels for Expectation Maximization with Gaussian Mixture Models
 *
 * Author: Andrew Pangborn
 * 
 * Department of Computer Engineering
 * Rochester Institute of Technology
 */


#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gaussian.h"

/*
 * Compute the multivariate mean of the FCS data
 */ 
void mvtmeans(float* fcs_data, int num_dimensions, int num_events, float* means,
              sycl::nd_item<3> item_ct1) {
    int tid = item_ct1.get_local_id(2);

    if(tid < num_dimensions) {
        means[tid] = 0.0;

        // Sum up all the values for each dimension
        for(int i = 0; i < num_events; i++) {
            means[tid] += fcs_data[i*num_dimensions+tid];
        }

        // Divide by the # of elements to get the average
        means[tid] /= (float) num_events;
    }
}

void averageVariance(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar,
                     sycl::nd_item<3> item_ct1, float *variances,
                     float *total_variance) {
    int tid = item_ct1.get_local_id(2);

    // Compute average variance for each dimension
    if(tid < num_dimensions) {
        variances[tid] = 0.0;
        // Sum up all the variance
        for(int i = 0; i < num_events; i++) {
            // variance = (data - mean)^2
            variances[tid] += (fcs_data[i*num_dimensions + tid])*(fcs_data[i*num_dimensions + tid]);
        }
        variances[tid] /= (float) num_events;
        variances[tid] -= means[tid]*means[tid];
    }

    item_ct1.barrier();

    if(tid == 0) {
        *total_variance = 0.0;
        for(int i=0; i<num_dimensions;i++) {
            *total_variance += variances[i];
        }
        *avgvar = *total_variance / (float)num_dimensions;
    }
}

// Inverts an NxN matrix 'data' stored as a 1D array in-place
// 'actualsize' is N
// Computes the log of the determinant of the origianl matrix in the process
void invert(float* data, int actualsize, float* log_determinant,
            sycl::nd_item<3> item_ct1)  {
    int maxsize = actualsize;
    int n = actualsize;

    if (item_ct1.get_local_id(2) == 0) {
        *log_determinant = 0.0;

      // sanity check        
      if (actualsize == 1) {
        *log_determinant = sycl::log(data[0]);
        data[0] = 1.0 / data[0];
      } else {

          for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
          for (int i=1; i < actualsize; i++)  { 
            for (int j=i; j < actualsize; j++)  { // do a column of L
              float sum = 0.0;
              for (int k = 0; k < i; k++)  
                  sum += data[j*maxsize+k] * data[k*maxsize+i];
              data[j*maxsize+i] -= sum;
              }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
              float sum = 0.0;
              for (int k = 0; k < i; k++)
                  sum += data[i*maxsize+k]*data[k*maxsize+j];
              data[i*maxsize+j] = 
                 (data[i*maxsize+j]-sum) / data[i*maxsize+i];
              }
            }
            
            for(int i=0; i<actualsize; i++) {
                *log_determinant += sycl::log(sycl::fabs(data[i * n + i]));
            }
            
          for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
              float x = 1.0;
              if ( i != j ) {
                x = 0.0;
                for ( int k = i; k < j; k++ ) 
                    x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
              data[j*maxsize+i] = x / data[j*maxsize+j];
              }
          for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
              if ( i == j ) continue;
              float sum = 0.0;
              for ( int k = i; k < j; k++ )
                  sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
              data[i*maxsize+j] = -sum;
              }
          for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
              float sum = 0.0;
              for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                  sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
              data[j*maxsize+i] = sum;
              }
        }
    }
 }


void compute_pi(clusters_t* clusters, int num_clusters,
                sycl::nd_item<3> item_ct1, float *sum) {

    if (item_ct1.get_local_id(2) == 0) {
        *sum = 0.0;
        for(int i=0; i<num_clusters; i++) {
            *sum += clusters->N[i];
        }
    }

    item_ct1.barrier();

    for (int c = item_ct1.get_local_id(2); c < num_clusters;
         c += item_ct1.get_local_range().get(2)) {
        if(clusters->N[c] < 0.5f) {
            clusters->pi[item_ct1.get_local_id(2)] = 1e-10;
        } else {
            clusters->pi[item_ct1.get_local_id(2)] = clusters->N[c] / *sum;
        }
    }

    item_ct1.barrier();
}


void compute_constants(clusters_t* clusters, int num_clusters, int num_dimensions,
                       sycl::nd_item<3> item_ct1, float *determinant_arg,
                       float *matrix) {
    int tid = item_ct1.get_local_id(2);
    int num_threads = item_ct1.get_local_range().get(2);
    int num_elements = num_dimensions*num_dimensions;

     // only one thread computes the inverse so we need a shared argument

    float log_determinant;

    // Invert the matrix for every cluster
    int c = item_ct1.get_group(2);
    // Copy the R matrix into shared memory for doing the matrix inversion
    for(int i=tid; i<num_elements; i+= num_threads ) {
        matrix[i] = clusters->R[c*num_dimensions*num_dimensions+i];
    }

    item_ct1.barrier();
#if DIAG_ONLY
        if(tid == 0) { 
            determinant_arg = 1.0f;
            for(int i=0; i < num_dimensions; i++) {
                determinant_arg *= matrix[i*num_dimensions+i];
                matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];
            }
            determinant_arg = logf(determinant_arg);
        }
    #else
        invert(matrix, num_dimensions, determinant_arg, item_ct1);
    #endif
    item_ct1.barrier();
    log_determinant = *determinant_arg;

    // Copy the matrx from shared memory back into the cluster memory
    for(int i=tid; i<num_elements; i+= num_threads) {
        clusters->Rinv[c*num_dimensions*num_dimensions+i] = matrix[i];
    }

    item_ct1.barrier();

    // Compute the constant
    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
    // This constant is used in all E-step likelihood calculations
    if(tid == 0) {
        clusters->constant[c] =
            -num_dimensions * 0.5f * sycl::log((float)(2.0f * PI)) -
            0.5f * log_determinant;
    }
}

/*
 * Computes the constant, pi, Rinv for each cluster
 * 
 * Needs to be launched with the number of blocks = number of clusters
 */
SYCL_EXTERNAL void constants_kernel(clusters_t *clusters, int num_clusters,
                                    int num_dimensions,
                                    sycl::nd_item<3> item_ct1,
                                    float *determinant_arg, float *sum,
                                    float *matrix) {

    // compute_constants(clusters,num_clusters,num_dimensions);

    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);
    int num_threads = item_ct1.get_local_range().get(2);
    int num_elements = num_dimensions*num_dimensions;

     // only one thread computes the inverse so we need a shared argument

    float log_determinant;
    
    
    // Invert the matrix for every cluster

    // Copy the R matrix into shared memory for doing the matrix inversion
    for(int i=tid; i<num_elements; i+= num_threads ) {
        matrix[i] = clusters->R[bid*num_dimensions*num_dimensions+i];
    }

    item_ct1.barrier();
#if DIAG_ONLY
        if(tid == 0) { 
            determinant_arg = 1.0f;
            for(int i=0; i < num_dimensions; i++) {
                determinant_arg *= matrix[i*num_dimensions+i];
                matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];
            }
            determinant_arg = logf(determinant_arg);
        }
    #else
        invert(matrix, num_dimensions, determinant_arg, item_ct1);
    #endif
    item_ct1.barrier();
    log_determinant = *determinant_arg;

    // Copy the matrx from shared memory back into the cluster memory
    for(int i=tid; i<num_elements; i+= num_threads) {
        clusters->Rinv[bid*num_dimensions*num_dimensions+i] = matrix[i];
    }

    item_ct1.barrier();

    // Compute the constant
    // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
    // This constant is used in all E-step likelihood calculations
    if(tid == 0) {
        clusters->constant[bid] =
            -num_dimensions * 0.5f * sycl::log((float)(2.0f * PI)) -
            0.5f * log_determinant;
    }

    item_ct1.barrier();

    if(bid == 0) {
        // compute_pi(clusters,num_clusters);
      
      if(tid == 0) {
          *sum = 0.0;
          for(int i=0; i<num_clusters; i++) {
              *sum += clusters->N[i];
          }
      }

      item_ct1.barrier();

      for(int i = tid; i < num_clusters; i += num_threads) {
          if(clusters->N[i] < 0.5f) {
              clusters->pi[tid] = 1e-10;
          } else {
              clusters->pi[tid] = clusters->N[i] / *sum;
          }
      }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! @param fcs_data         FCS data: [num_events]
//! @param clusters         Clusters: [num_clusters]
//! @param num_dimensions   number of dimensions in an FCS event
//! @param num_events       number of FCS events
////////////////////////////////////////////////////////////////////////////////
SYCL_EXTERNAL void seed_clusters_kernel(
    const float *fcs_data, clusters_t *clusters, const int num_dimensions,
    const int num_clusters, const int num_events, sycl::nd_item<3> item_ct1,
    float *means, float *avgvar, float *variances, float *total_variance)
{
    int tid = item_ct1.get_local_id(2);
    int num_threads = item_ct1.get_local_range().get(2);
    int row, col;
    float seed;
        
    // Number of elements in the covariance matrix
    int num_elements = num_dimensions*num_dimensions; 

    // shared memory

    // Compute the means
    // mvtmeans(fcs_data, num_dimensions, num_events, means);

    if(tid < num_dimensions) {
        means[tid] = 0.0;

        // Sum up all the values for each dimension
        for(int i = 0; i < num_events; i++) {
            means[tid] += fcs_data[i*num_dimensions+tid];
        }

        // Divide by the # of elements to get the average
        means[tid] /= (float) num_events;
    }

    item_ct1.barrier();

    // Compute the average variance
    // averageVariance(fcs_data, means, num_dimensions, num_events, &avgvar);
    
    // Compute average variance for each dimension
    if(tid < num_dimensions) {
        variances[tid] = 0.0;
        // Sum up all the variance
        for(int i = 0; i < num_events; i++) {
            // variance = (data - mean)^2
            variances[tid] += (fcs_data[i*num_dimensions + tid])*(fcs_data[i*num_dimensions + tid]);
        }
        variances[tid] /= (float) num_events;
        variances[tid] -= means[tid]*means[tid];
    }

    item_ct1.barrier();

    if(tid == 0) {
        *total_variance = 0.0;
        for(int i=0; i<num_dimensions;i++) {
            *total_variance += variances[i];
        }
        *avgvar = *total_variance / (float)num_dimensions;
    }

    item_ct1.barrier();

    if(num_clusters > 1) {
        seed = (num_events-1.0f)/(num_clusters-1.0f);
    } else {
        seed = 0.0;
    }
    
    // Seed the pi, means, and covariances for every cluster
    for(int c=0; c < num_clusters; c++) {
        if(tid < num_dimensions) {
            clusters->means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
        }
          
        for(int i=tid; i < num_elements; i+= num_threads) {
            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
            row = (i) / num_dimensions;
            col = (i) % num_dimensions;

            if(row == col) {
                clusters->R[c*num_dimensions*num_dimensions+i] = 1.0f;
            } else {
                clusters->R[c*num_dimensions*num_dimensions+i] = 0.0f;
            }
        }
        if(tid == 0) {
            clusters->pi[c] = 1.0f/((float)num_clusters);
            clusters->N[c] = ((float) num_events) / ((float)num_clusters);
            clusters->avgvar[c] = *avgvar / COVARIANCE_DYNAMIC_RANGE;
        }
    }
}

float parallelSum(float* data, const unsigned int ndata,
                  sycl::nd_item<3> item_ct1) {
  const unsigned int tid = item_ct1.get_local_id(2);
  float t;

  item_ct1.barrier();

  // Butterfly sum.  ndata MUST be a power of 2.
  for(unsigned int bit = ndata >> 1; bit > 0; bit >>= 1) {
    t = data[tid] + data[tid ^ bit]; item_ct1.barrier();
    data[tid] = t; item_ct1.barrier();
  }
  return data[tid];
}

void compute_indices(int num_events, int* start, int* stop,
                     sycl::nd_item<3> item_ct1) {
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / NUM_BLOCKS;
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

    *start =
        item_ct1.get_group(1) * num_pixels_per_block + item_ct1.get_local_id(2);

    // Last block will handle the leftover events
    if (item_ct1.get_group(1) == item_ct1.get_group_range(1) - 1) {
        *stop = num_events;
    } else {
        *stop = (item_ct1.get_group(1) + 1) * num_pixels_per_block;
    }
}

SYCL_EXTERNAL void estep1(float *data, clusters_t *clusters, int num_dimensions,
                          int num_events, sycl::nd_item<3> item_ct1,
                          float *means, float *Rinv) {

    // Cached cluster parameters

    float cluster_pi;
    float constant;
    const unsigned int tid = item_ct1.get_local_id(2);

    int start_index;
    int end_index;

    int c = item_ct1.get_group(2);

    compute_indices(num_events, &start_index, &end_index, item_ct1);

    float like;

    // This loop computes the expectation of every event into every cluster
    //
    // P(k|n) = L(x_n|mu_k,R_k)*P(k) / P(x_n)
    //
    // Compute log-likelihood for every cluster for each event
    // L = constant*exp(-0.5*(x-mu)*Rinv*(x-mu))
    // log_L = log_constant - 0.5*(x-u)*Rinv*(x-mu)
    // the constant stored in clusters[c].constant is already the log of the constant
    
    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }

    // copy the covariance inverse into shared memory
    for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {
        Rinv[i] = clusters->Rinv[c*num_dimensions*num_dimensions+i]; 
    }
    
    cluster_pi = clusters->pi[c];
    constant = clusters->constant[c];

    // Sync to wait for all params to be loaded to shared memory
    item_ct1.barrier();

    for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {
       like = 0.0f;
        // this does the loglikelihood calculation
        #if DIAG_ONLY
            for(int j=0; j<num_dimensions; j++) {
                like += (data[j*num_events+event]-means[j]) * (data[j*num_events+event]-means[j]) * Rinv[j*num_dimensions+j];
            }
        #else
            for(int i=0; i<num_dimensions; i++) {
                for(int j=0; j<num_dimensions; j++) {
                    like += (data[i*num_events+event]-means[i]) * (data[j*num_events+event]-means[j]) * Rinv[i*num_dimensions+j];
                }
            }
        #endif
        // numerator of the E-step probability computation
        clusters->memberships[c * num_events + event] =
            -0.5f * like + constant + sycl::log(cluster_pi);
    }
}

SYCL_EXTERNAL void estep2(float *fcs_data, clusters_t *clusters,
                          int num_dimensions, int num_clusters, int num_events,
                          float *likelihood, sycl::nd_item<3> item_ct1,
                          float *total_likelihoods) {
    float temp;
    float thread_likelihood = 0.0f;

    float max_likelihood;
    float denominator_sum;
    
    // Break up the events evenly between the blocks
    int num_pixels_per_block = num_events / item_ct1.get_group_range(2);
    // Make sure the events being accessed by the block are aligned to a multiple of 16
    num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);
    int tid = item_ct1.get_local_id(2);

    int start_index;
    int end_index;
    start_index = item_ct1.get_group(2) * num_pixels_per_block + tid;

    // Last block will handle the leftover events
    if (item_ct1.get_group(2) == item_ct1.get_group_range(2) - 1) {
        end_index = num_events;
    } else {
        end_index = (item_ct1.get_group(2) + 1) * num_pixels_per_block;
    }
    
    total_likelihoods[tid] = 0.0;

    // P(x_n) = sum of likelihoods weighted by P(k) (their probability, cluster[c].pi)
    //  log(a+b) != log(a) + log(b) so we need to do the log of the sum of the exponentials

    //  For the sake of numerical stability, we first find the max and scale the values
    //  That way, the maximum value ever going into the exp function is 0 and we avoid overflow

    //  log-sum-exp formula:
    //  log(sum(exp(x_i)) = max(z) + log(sum(exp(z_i-max(z))))
    for(int pixel=start_index; pixel<end_index; pixel += NUM_THREADS_ESTEP) {
        // find the maximum likelihood for this event
        max_likelihood = clusters->memberships[pixel];
        for(int c=1; c<num_clusters; c++) {
            max_likelihood = sycl::fmax(
                max_likelihood, clusters->memberships[c * num_events + pixel]);
        }

        // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
        denominator_sum = 0.0;
        for(int c=0; c<num_clusters; c++) {
            temp = sycl::exp(clusters->memberships[c * num_events + pixel] -
                             max_likelihood);
            denominator_sum += temp;
        }
        denominator_sum = max_likelihood + sycl::log(denominator_sum);
        thread_likelihood += denominator_sum;
        
        // Divide by denominator, also effectively normalize probabilities
        // exp(log(p) - log(denom)) == p / denom
        for(int c=0; c<num_clusters; c++) {
            clusters->memberships[c * num_events + pixel] =
                sycl::exp(clusters->memberships[c * num_events + pixel] -
                          denominator_sum);
            //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters->memberships[c*num_events+pixel]);
        }
    }
    
    total_likelihoods[tid] = thread_likelihood;
    item_ct1.barrier();

    temp = parallelSum(total_likelihoods, NUM_THREADS_ESTEP, item_ct1);
    if(tid == 0) {
        likelihood[item_ct1.get_group(2)] = temp;
    }
}


/*
 * Means kernel
 * MultiGPU version, sums up all of the elements, but does not divide by N. 
 * This task is left for the host after combing results from multiple GPUs
 *
 * Should be launched with [M x D] grid
 */
SYCL_EXTERNAL void mstep_means(float *fcs_data, clusters_t *clusters,
                               int num_dimensions, int num_clusters,
                               int num_events, sycl::nd_item<3> item_ct1,
                               float *temp_sum) {
    // One block per cluster, per dimension:  (M x D) grid of blocks
    int tid = item_ct1.get_local_id(2);
    int num_threads = item_ct1.get_local_range().get(2);
    int c = item_ct1.get_group(2); // cluster number
    int d = item_ct1.get_group(1); // dimension number

    float sum = 0.0f;
    
    for(int event=tid; event < num_events; event+= num_threads) {
        sum += fcs_data[d*num_events+event]*clusters->memberships[c*num_events+event];
    }
    temp_sum[tid] = sum;

    item_ct1.barrier();

    // Reduce partial sums
    sum = parallelSum(temp_sum, NUM_THREADS_MSTEP, item_ct1);
    if(tid == 0) {
        clusters->means[c*num_dimensions+d] = sum;
    }

    /*if(tid == 0) {
        for(int i=1; i < num_threads; i++) {
            temp_sum[0] += temp_sum[i];
        }
        clusters->means[c*num_dimensions+d] = temp_sum[0];
        //clusters->means[c*num_dimensions+d] = temp_sum[0] / clusters->N[c];
    }*/
}

/*
 * Computes the size of each cluster, N
 * Should be launched with M blocks (where M = number of clusters)
 */
SYCL_EXTERNAL void mstep_N(clusters_t *clusters, int num_dimensions,
                           int num_clusters, int num_events,
                           sycl::nd_item<3> item_ct1, float *temp_sums) {

    int tid = item_ct1.get_local_id(2);
    int num_threads = item_ct1.get_local_range().get(2);
    int c = item_ct1.get_group(2);

    // Need to store the sum computed by each thread so in the end
    // a single thread can reduce to get the final sum

    // Compute new N
    float sum = 0.0f;
    // Break all the events accross the threads, add up probabilities
    for(int event=tid; event < num_events; event += num_threads) {
        sum += clusters->memberships[c*num_events+event];
    }
    temp_sums[tid] = sum;

    item_ct1.barrier();

    sum = parallelSum(temp_sums, NUM_THREADS_MSTEP, item_ct1);
    if(tid == 0) {
        clusters->N[c] = sum;
        clusters->pi[c] = sum;
    }
    
    // Let the first thread add up all the intermediate sums
    // Could do a parallel reduction...doubt it's really worth it for so few elements though
    /*if(tid == 0) {
        clusters->N[c] = 0.0;
        for(int j=0; j<num_threads; j++) {
            clusters->N[c] += temp_sums[j];
        }
        //printf("clusters[%d].N = %f\n",c,clusters[c].N);
        
        // Set PI to the # of expected items, and then normalize it later
        clusters->pi[c] = clusters->N[c];
    }*/
}
   
/*
 * Computes the row and col of a square matrix based on the index into
 * a lower triangular (with diagonal) matrix
 * 
 * Used to determine what row/col should be computed for covariance
 * based on a block index.
 */
void compute_row_col(int n, int* row, int* col, sycl::nd_item<3> item_ct1) {
    int i = 0;
    for(int r=0; r < n; r++) {
        for(int c=0; c <= r; c++) {
            if (i == item_ct1.get_group(1)) {
                *row = r;
                *col = c;
                return;
            }
            i++;
        }
    }
}
 
/*
 * Computes the covariance matrices of the data (R matrix)
 * Must be launched with a M x D*D grid of blocks: 
 *  i.e. dim3 gridDim(num_clusters,num_dimensions*num_dimensions)
 */
void
mstep_covariance1(float* fcs_data, clusters_t* clusters, int num_dimensions, int num_clusters, int num_events,
                  sycl::nd_item<3> item_ct1, float *means, float *temp_sums) {
    int tid =
        item_ct1.get_local_id(2); // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c;
    compute_row_col(num_dimensions, &row, &col, item_ct1);
    //row = blockIdx.y / num_dimensions;
    //col = blockIdx.y % num_dimensions;

    item_ct1.barrier();

    c = item_ct1.get_group(2); // Determines what cluster this block is handling

    int matrix_index = row * num_dimensions + col;

    #if DIAG_ONLY
    if(row != col) {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = 0.0;
        return;
    }
    #endif 

    // Store the means in shared memory to speed up the covariance computations

    // copy the means for this cluster into shared memory
    if(tid < num_dimensions) {
        means[tid] = clusters->means[c*num_dimensions+tid];
    }

    // Sync to wait for all params to be loaded to shared memory
    item_ct1.barrier();

    float cov_sum = 0.0;

    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
        cov_sum += (fcs_data[row*num_events+event]-means[row])*
                   (fcs_data[col*num_events+event]-means[col])*clusters->memberships[c*num_events+event]; 
    }
    temp_sums[tid] = cov_sum;

    item_ct1.barrier();

    cov_sum = parallelSum(temp_sums, NUM_THREADS_MSTEP, item_ct1);

    if(tid == 0) {
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        // Set the symmetric value
        matrix_index = col*num_dimensions+row;
        clusters->R[c*num_dimensions*num_dimensions+matrix_index] = cov_sum;
        
        // Regularize matrix - adds some variance to the diagonal elements
        // Helps keep covariance matrix non-singular (so it can be inverted)
        // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined at top of this file
        if(row == col) {
            clusters->R[c*num_dimensions*num_dimensions+matrix_index] += clusters->avgvar[c];
        }
    }
}

SYCL_EXTERNAL void mstep_covariance2(float *fcs_data, clusters_t *clusters,
                                     int num_dimensions, int num_clusters,
                                     int num_events, sycl::nd_item<3> item_ct1,
                                     float *means_row, float *means_col,
                                     float *temp_sums) {
    int tid =
        item_ct1.get_local_id(2); // easier variable name for our thread ID

    // Determine what row,col this matrix is handling, also handles the symmetric element
    int row,col,c1;
    compute_row_col(num_dimensions, &row, &col, item_ct1);

    item_ct1.barrier();

    c1 = item_ct1.get_group(2) *
         NUM_CLUSTERS_PER_BLOCK; // Determines what cluster this block is
                                 // handling

#if DIAG_ONLY
    if(row != col) {
        clusters->R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;
        clusters->R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;
        return;
    }
    #endif 

    // Store the means in shared memory to speed up the covariance computations

    //if(tid < NUM_CLUSTERS_PER_BLOCK) {
    if ((tid < sycl::min(num_clusters, NUM_CLUSTERS_PER_BLOCK)) // c1 = 0
        && (c1 + tid < num_clusters)) {
        means_row[tid] = clusters->means[(c1+tid)*num_dimensions+row];
        means_col[tid] = clusters->means[(c1+tid)*num_dimensions+col];
    }

    // Sync to wait for all params to be loaded to shared memory
    item_ct1.barrier();

    // 256 * 6

    float cov_sum1 = 0.0f;
    float cov_sum2 = 0.0f;
    float cov_sum3 = 0.0f;
    float cov_sum4 = 0.0f;
    float cov_sum5 = 0.0f;
    float cov_sum6 = 0.0f;
    float val1,val2;
        
    for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
       temp_sums[c*NUM_THREADS_MSTEP+tid] = 0.0;
    } 

    for(int event=tid; event < num_events; event+=NUM_THREADS_MSTEP) {
        val1 = fcs_data[row*num_events+event];
        val2 = fcs_data[col*num_events+event];
        cov_sum1 += (val1-means_row[0])*(val2-means_col[0])*clusters->memberships[c1*num_events+event]; 
        cov_sum2 += (val1-means_row[1])*(val2-means_col[1])*clusters->memberships[(c1+1)*num_events+event]; 
        cov_sum3 += (val1-means_row[2])*(val2-means_col[2])*clusters->memberships[(c1+2)*num_events+event]; 
        cov_sum4 += (val1-means_row[3])*(val2-means_col[3])*clusters->memberships[(c1+3)*num_events+event]; 
        cov_sum5 += (val1-means_row[4])*(val2-means_col[4])*clusters->memberships[(c1+4)*num_events+event]; 
        cov_sum6 += (val1-means_row[5])*(val2-means_col[5])*clusters->memberships[(c1+5)*num_events+event]; 
    }
    temp_sums[0*NUM_THREADS_MSTEP+tid] = cov_sum1;
    temp_sums[1*NUM_THREADS_MSTEP+tid] = cov_sum2;
    temp_sums[2*NUM_THREADS_MSTEP+tid] = cov_sum3;
    temp_sums[3*NUM_THREADS_MSTEP+tid] = cov_sum4;
    temp_sums[4*NUM_THREADS_MSTEP+tid] = cov_sum5;
    temp_sums[5*NUM_THREADS_MSTEP+tid] = cov_sum6;

    item_ct1.barrier();

    for(int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
        temp_sums[c * NUM_THREADS_MSTEP + tid] = parallelSum(
            &temp_sums[c * NUM_THREADS_MSTEP], NUM_THREADS_MSTEP, item_ct1);
        item_ct1.barrier();
    }
    
    if(tid == 0) {
        for(int c=0; c < NUM_CLUSTERS_PER_BLOCK && (c+c1) < num_clusters; c++) {
            int offset = (c+c1)*num_dimensions*num_dimensions;
            cov_sum1 = temp_sums[c*NUM_THREADS_MSTEP];
            clusters->R[offset+row*num_dimensions+col] = cov_sum1;
            // Set the symmetric value
            clusters->R[offset+col*num_dimensions+row] = cov_sum1;
            
            // Regularize matrix - adds some variance to the diagonal elements
            // Helps keep covariance matrix non-singular (so it can be inverted)
            // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined in gaussian.h
            if(row == col) {
                clusters->R[offset+row*num_dimensions+col] += clusters->avgvar[c+c1];
            }
        }
    }
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
