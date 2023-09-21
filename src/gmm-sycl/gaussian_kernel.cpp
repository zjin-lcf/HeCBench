/*
 * Kernels for Expectation Maximization with Gaussian Mixture Models
 *
 */


#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include "gaussian.h"


// Inverts an NxN matrix 'data' stored as a 1D array in-place
// 'actualsize' is N
// Computes the log of the determinant of the origianl matrix in the process
void invert(float* data, int actualsize, float &log_determinant)  {
  int maxsize = actualsize;
  int n = actualsize;

  log_determinant = 0.0;

  // sanity check        
  if (actualsize == 1) {
    log_determinant = sycl::log(data[0]);
    data[0] = 1.f / data[0];
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
      log_determinant += sycl::log(sycl::fabs(data[i*n+i]));
    }

    for ( int i = 0; i < actualsize; i++ )  // invert L
      for ( int j = i; j < actualsize; j++ )  {
        float x = 1.f;
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
          sum += data[k*maxsize+j]*( (i==k) ? 1.f : data[i*maxsize+k] );
        data[i*maxsize+j] = -sum;
      }
    for ( int i = 0; i < actualsize; i++ )   // final inversion
      for ( int j = 0; j < actualsize; j++ )  {
        float sum = 0.0;
        for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
          sum += ((j==k)?1.f:data[j*maxsize+k])*data[k*maxsize+i];
        data[j*maxsize+i] = sum;
      }
  }
}

/*
 * Computes the row and col of a square matrix based on the index into
 * a lower triangular (with diagonal) matrix
 * 
 * Used to determine what row/col should be computed for covariance
 * based on a block index.
 */
void compute_row_col(sycl::nd_item<2> &item, const int n, int* row, int* col) {
  int i = 0;
  for(int r=0; r < n; r++) {
    for(int c=0; c <= r; c++) {
      if(i == item.get_group(0)) {  
        *row = r;
        *col = c;
        return;
      }
      i++;
    }
  }
}

/*
 * Computes the constant, pi, Rinv for each cluster
 * 
 * Needs to be launched with the number of blocks = number of clusters
 */
void constants_kernel(
    sycl::nd_item<1> &item, 
    const float *clusters_R,
    float *clusters_Rinv,
    const float *clusters_N,
    float *clusters_pi,
    float *clusters_constant,
    float *clusters_avgvar,
    float *matrix,
    float &determinant_arg,
    float &sum,
    const int num_clusters, 
    const int num_dimensions) {

  // compute_constants(clusters,num_clusters,num_dimensions);

  int tid = item.get_local_id(0);
  int bid = item.get_group(0); 
  int num_threads = item.get_local_range(0);
  int num_elements = num_dimensions*num_dimensions;

  float log_determinant;


  // Invert the matrix for every cluster

  // Copy the R matrix into shared memory for doing the matrix inversion
  for(int i=tid; i<num_elements; i+= num_threads ) {
    matrix[i] = clusters_R[bid*num_dimensions*num_dimensions+i];
  }

  item.barrier(sycl::access::fence_space::local_space); 

  if(tid == 0) { 
#if DIAG_ONLY
    determinant_arg = 1.0f;
    for(int i=0; i < num_dimensions; i++) {
      determinant_arg *= matrix[i*num_dimensions+i];
      matrix[i*num_dimensions+i] = 1.0f / matrix[i*num_dimensions+i];
    }
    determinant_arg = sycl::log(determinant_arg);
#else 
  invert(matrix,num_dimensions,determinant_arg);
#endif
  }
  item.barrier(sycl::access::fence_space::local_space); 
  log_determinant = determinant_arg;

  // Copy the matrx from shared memory back into the cluster memory
  for(int i=tid; i<num_elements; i+= num_threads) {
    clusters_Rinv[bid*num_dimensions*num_dimensions+i] = matrix[i];
  }

  item.barrier(sycl::access::fence_space::local_space);

  // Compute the constant
  // Equivilent to: log(1/((2*PI)^(M/2)*det(R)^(1/2)))
  // This constant is used in all E-step likelihood calculations
  if(tid == 0) {
    clusters_constant[bid] = -num_dimensions*0.5f*sycl::log(2.0f*PI) - 0.5f*log_determinant;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if(bid == 0) {
    // compute_pi(clusters,num_clusters);

    if(tid == 0) {
      sum = 0.0;
      for(int i=0; i<num_clusters; i++) {
        sum += clusters_N[i];
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    for(int i = tid; i < num_clusters; i += num_threads) {
      if(clusters_N[i] < 0.5f) {
        clusters_pi[tid] = 1e-10;
      } else {
        clusters_pi[tid] = clusters_N[i] / sum;
      }
    }
  }
}

void estep1_kernel(
    sycl::nd_item<2> &item,
    const float *data, 
    const float* clusters_Rinv, 
    float *clusters_memberships, 
    const float *clusters_pi, 
    const float *clusters_constant,
    const float *clusters_means, 
    float *means,
    float *Rinv,
    const int num_dimensions, 
    const int num_events) 
{
  int c = item.get_group(1);

  // compute_indices(num_events,&start_index,&end_index);

  // Break up the events evenly between the blocks
  int num_pixels_per_block = num_events / NUM_BLOCKS;
  // Make sure the events being accessed by the block are aligned to a multiple of 16
  num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

  const unsigned int tid = item.get_local_id(1);
  const unsigned int groupIdx_y = item.get_group(0);
  const unsigned int groupDim_y = item.get_group_range(0);

  int start_index = groupIdx_y * num_pixels_per_block + tid;

  // Last block will handle the leftover events
  int end_index;
  if(groupIdx_y == groupDim_y-1) {
    end_index = num_events;
  } else { 
    end_index = (groupIdx_y+1) * num_pixels_per_block;
  }

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
    means[tid] = clusters_means[c*num_dimensions+tid];
  }

  // copy the covariance inverse into shared memory
  for(int i=tid; i < num_dimensions*num_dimensions; i+= NUM_THREADS_ESTEP) {
    Rinv[i] = clusters_Rinv[c*num_dimensions*num_dimensions+i]; 
  }

  float cluster_pi = clusters_pi[c];
  float constant = clusters_constant[c];

  // Sync to wait for all params to be loaded to shared memory
  item.barrier(sycl::access::fence_space::local_space);

  for(int event=start_index; event<end_index; event += NUM_THREADS_ESTEP) {
    float like = 0.0f;
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
    clusters_memberships[c*num_events+event] = -0.5f * like + constant + sycl::log(cluster_pi);
  }
}

void estep2_kernel(
    sycl::nd_item<1> &item,
    float* clusters_memberships,
    float* likelihood,
    float* total_likelihoods, 
    const int num_dimensions, 
    const int num_clusters, 
    const int num_events ) 
{
  float temp;
  float thread_likelihood = 0.0f;
  float max_likelihood;
  float denominator_sum;

  int groups = item.get_group_range(0);
  int tid = item.get_local_id(0);
  int bid = item.get_group(0);

  // Break up the events evenly between the blocks
  int num_pixels_per_block = num_events / groups; //gridDim.x;
  // Make sure the events being accessed by the block are aligned to a multiple of 16
  num_pixels_per_block = num_pixels_per_block - (num_pixels_per_block % 16);

  int start_index = bid * num_pixels_per_block + tid;

  // Last block will handle the leftover events
  int end_index;
  if(bid == groups-1) {
    end_index = num_events;
  } else {
    end_index = (bid+1) * num_pixels_per_block;
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
    max_likelihood = clusters_memberships[pixel];
    for(int c=1; c<num_clusters; c++) {
      max_likelihood = sycl::fmax(max_likelihood,clusters_memberships[c*num_events+pixel]);
    }

    // Compute P(x_n), the denominator of the probability (sum of weighted likelihoods)
    denominator_sum = 0.0;
    for(int c=0; c<num_clusters; c++) {
      temp = sycl::exp(clusters_memberships[c*num_events+pixel]-max_likelihood);
      denominator_sum += temp;
    }
    denominator_sum = max_likelihood + sycl::log(denominator_sum);
    thread_likelihood += denominator_sum;

    // Divide by denominator, also effectively normalize probabilities
    // exp(log(p) - log(denom)) == p / denom
    for(int c=0; c<num_clusters; c++) {
      clusters_memberships[c*num_events+pixel] = 
        sycl::exp(clusters_memberships[c*num_events+pixel] - denominator_sum);
      //printf("Probability that pixel #%d is in cluster #%d: %f\n",pixel,c,clusters->memberships[c*num_events+pixel]);
    }
  }

  total_likelihoods[tid] = thread_likelihood;
  item.barrier(sycl::access::fence_space::local_space);

  // temp = parallelSum(total_likelihoods,NUM_THREADS_ESTEP);
  for (unsigned int bit = NUM_THREADS_ESTEP >> 1; bit > 0; bit >>= 1) {
    float t = total_likelihoods[tid] + total_likelihoods[tid^bit];
    item.barrier(sycl::access::fence_space::local_space);
    total_likelihoods[tid] = t;
    item.barrier(sycl::access::fence_space::local_space);
  }

  if(tid == 0) {
    likelihood[bid] = total_likelihoods[tid];
  }
}



#endif // #ifndef _TEMPLATE_KERNEL_H_
