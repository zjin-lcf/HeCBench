#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
/*
 * Inverts a square matrix (stored as a 1D float array)
 *
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
  int maxsize = actualsize;
  int n = actualsize;
  *log_determinant = 0.0;

  if (actualsize == 1) { // special case, dimensionality == 1
    *log_determinant = ::logf(data[0]);
    data[0] = 1.0 / data[0];
  } else if(actualsize >= 2) { // dimensionality >= 2
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
      *log_determinant += ::log10(fabs(data[i*n+i]));
      //printf("log_determinant: %e\n",*log_determinant); 
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

  } else {
    PRINT("Error: Invalid dimensionality for invert(...)\n");
  }
}

///////////////////////////////////////////////////////////////////////////////
// Validate command line arguments
///////////////////////////////////////////////////////////////////////////////
int validateArguments(int argc, char** argv, int* num_clusters, int* target_num_clusters) {
  if(argc <= 5 && argc >= 4) {
    // parse num_clusters
    if(!sscanf(argv[1],"%d",num_clusters)) {
      printf("Invalid number of starting clusters\n\n");
      printUsage(argv);
      return 1;
    } 

    // Check bounds for num_clusters
    if(*num_clusters < 1) {
      printf("Invalid number of starting clusters\n\n");
      printUsage(argv);
      return 1;
    }

    // parse infile
    FILE* infile = fopen(argv[2],"r");
    if(!infile) {
      printf("Invalid infile.\n\n");
      printUsage(argv);
      return 2;
    } 

    // parse target_num_clusters
    if(argc == 5) {
      if(!sscanf(argv[4],"%d",target_num_clusters)) {
        printf("Invalid number of desired clusters.\n\n");
        printUsage(argv);
        return 4;
      }
      if(*target_num_clusters > *num_clusters) {
        printf("target_num_clusters must be less than equal to num_clusters\n\n");
        printUsage(argv);
        return 4;
      }
    } else {
      *target_num_clusters = 0;
    }

    // Clean up so the EPA is happy
    fclose(infile);
    return 0;
  } else {
    printUsage(argv);
    return 1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Print usage statement
///////////////////////////////////////////////////////////////////////////////
void printUsage(char** argv)
{
  printf("Usage: %s num_clusters infile outfile [target_num_clusters]\n",argv[0]);
  printf("\t num_clusters: The number of starting clusters\n");
  printf("\t infile: ASCII space-delimited FCS data file\n");
  printf("\t outfile: Clustering results output file\n");
  printf("\t target_num_clusters: A desired number of clusters. Must be less than or equal to num_clusters\n");
}

void writeCluster(FILE* f, clusters_t &clusters, const int c, const int num_dimensions) {
  fprintf(f,"Probability: %f\n", clusters.pi[c]);
  fprintf(f,"N: %f\n",clusters.N[c]);
  fprintf(f,"Means: ");
  for(int i=0; i<num_dimensions; i++){
    fprintf(f,"%f ",clusters.means[c*num_dimensions+i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\nR Matrix:\n");
  for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
      fprintf(f,"%f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
    }
    fprintf(f,"\n");
  }
  fflush(f);   
}

/*
 * Seeds the cluster centers (means) with random data points
 */
void seed_clusters(clusters_t* clusters, float* fcs_data, int num_clusters, int num_dimensions, int num_events) {
  float fraction;
  int seed;
  if(num_clusters > 1) {
    fraction = (num_events-1.0f)/(num_clusters-1.0f);
  } else {
    fraction = 0.0;
  }
  srand((unsigned int) time(NULL));
  // Sets the means from evenly distributed points in the input data
  for(int c=0; c < num_clusters; c++) {
    clusters->N[c] = (float)num_events/(float)num_clusters;
    for(int d=0; d < num_dimensions; d++)
      clusters->means[c*num_dimensions+d] = fcs_data[((int)(c*fraction))*num_dimensions+d];
  }
}

void add_clusters(clusters_t &clusters, const int c1, const int c2, clusters_t &temp_cluster, const int num_dimensions) {
  float wt1,wt2;

  wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
  wt2 = 1.0f - wt1;

  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
  }

  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from cluster1
      temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
          *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
          +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from cluster2
      temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
          *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
          +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
    }
  }

  // Compute pi
  temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];

  // compute N
  temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_cluster.constant[0] = (-num_dimensions)*0.5f*::logf(2.0f*PI)-0.5f*log_determinant;

  // avgvar same for all clusters
  temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t &dest, const int c_dest, clusters_t &src, const int c_src, const int num_dimensions) {
  dest.N[c_dest] = src.N[c_src];
  dest.pi[c_dest] = src.pi[c_src];
  dest.constant[c_dest] = src.constant[c_src];
  dest.avgvar[c_dest] = src.avgvar[c_src];
  memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}

void printCluster(clusters_t &clusters, const int c, const int num_dimensions) {
  writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t &clusters, const int c1, const int c2, clusters_t &temp_cluster, const int num_dimensions) {
  // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
  add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);

  return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}


// Free the cluster data structures on host
void freeCluster(clusters_t* c) {
  free(c->N);
  free(c->pi);
  free(c->constant);
  free(c->avgvar);
  free(c->means);
  free(c->R);
  free(c->Rinv);
  free(c->memberships);
}

// Free the cluster data structures on device
void freeClusterDevice(clusters_t *c) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  /*
  DPCT1003:0: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->N, q_ct1), 0));
  /*
  DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->pi, q_ct1), 0));
  /*
  DPCT1003:3: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->constant, q_ct1), 0));
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->avgvar, q_ct1), 0));
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->means, q_ct1), 0));
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->R, q_ct1), 0));
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->Rinv, q_ct1), 0));
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(c->memberships, q_ct1), 0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// Setup the cluster data structures on host
void setupCluster(clusters_t* c, const int num_clusters, const int num_events, const int num_dimensions) {
  c->N = (float*) malloc(sizeof(float)*num_clusters);
  c->pi = (float*) malloc(sizeof(float)*num_clusters);
  c->constant = (float*) malloc(sizeof(float)*num_clusters);
  c->avgvar = (float*) malloc(sizeof(float)*num_clusters);
  c->means = (float*) malloc(sizeof(float)*num_dimensions*num_clusters);
  c->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
  c->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*num_clusters);
  c->memberships = (float*) malloc(sizeof(float)*num_events*num_clusters);
}

// Setup the cluster data structures on device
clusters_t *setupClusterDevice(clusters_t *c, const int num_clusters,
                               const int num_events,
                               const int num_dimensions) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((c->N = sycl::malloc_device<float>(num_clusters, q_ct1), 0));
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((c->pi = sycl::malloc_device<float>(num_clusters, q_ct1), 0));
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (c->constant = sycl::malloc_device<float>(num_clusters, q_ct1), 0));
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (c->avgvar = sycl::malloc_device<float>(num_clusters, q_ct1), 0));
  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((c->means = (float *)sycl::malloc_device(
                      sizeof(float) * num_dimensions * num_clusters, q_ct1),
                  0));
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (c->R = (float *)sycl::malloc_device(sizeof(float) * num_dimensions *
                                               num_dimensions * num_clusters,
                                           q_ct1),
       0));
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (c->Rinv = (float *)sycl::malloc_device(sizeof(float) * num_dimensions *
                                                  num_dimensions * num_clusters,
                                              q_ct1),
       0));
  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((c->memberships = (float *)sycl::malloc_device(
                      sizeof(float) * num_events *
                          (num_clusters + NUM_CLUSTERS_PER_BLOCK -
                           num_clusters % NUM_CLUSTERS_PER_BLOCK),
                      q_ct1),
                  0));

  clusters_t* d_clusters;
  /*
  DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((d_clusters = sycl::malloc_device<clusters_t>(1, q_ct1), 0));

  // Copy Cluster data to device
  /*
  DPCT1003:18: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1.memcpy(d_clusters, c, sizeof(clusters_t)).wait(), 0));
  DEBUG("Finished copying cluster data to device.\n");
  return d_clusters;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void copyClusterFromDevice(clusters_t *c, clusters_t *c_tmp, clusters_t *d_c,
                           const int num_clusters,
                           const int num_dimensions) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  if (d_c != NULL)
    /*
    DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((q_ct1.memcpy(c_tmp, d_c, sizeof(clusters_t)).wait(), 0));
  // copy all of the arrays from the structs
  /*
  DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c->N, c_tmp->N, sizeof(float) * num_clusters).wait(), 0));
  /*
  DPCT1003:21: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c->pi, c_tmp->pi, sizeof(float) * num_clusters).wait(), 0));
  /*
  DPCT1003:22: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c->constant, c_tmp->constant, sizeof(float) * num_clusters)
           .wait(),
       0));
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c->avgvar, c_tmp->avgvar, sizeof(float) * num_clusters)
           .wait(),
       0));
  /*
  DPCT1003:24: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c->means, c_tmp->means,
                              sizeof(float) * num_dimensions * num_clusters)
                      .wait(),
                  0));
  /*
  DPCT1003:25: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c->R, c_tmp->R,
                              sizeof(float) * num_dimensions * num_dimensions *
                                  num_clusters)
                      .wait(),
                  0));
  /*
  DPCT1003:26: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c->Rinv, c_tmp->Rinv,
                              sizeof(float) * num_dimensions * num_dimensions *
                                  num_clusters)
                      .wait(),
                  0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void copyClusterToDevice(clusters_t *c, clusters_t *c_tmp,
                         const int num_clusters, const int num_dimensions) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
  /*
  DPCT1003:27: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c_tmp->N, c->N, sizeof(float) * num_clusters).wait(), 0));
  /*
  DPCT1003:28: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c_tmp->pi, c->pi, sizeof(float) * num_clusters).wait(), 0));
  /*
  DPCT1003:29: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c_tmp->constant, c->constant, sizeof(float) * num_clusters)
           .wait(),
       0));
  /*
  DPCT1003:30: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(c_tmp->avgvar, c->avgvar, sizeof(float) * num_clusters)
           .wait(),
       0));
  /*
  DPCT1003:31: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c_tmp->means, c->means,
                              sizeof(float) * num_dimensions * num_clusters)
                      .wait(),
                  0));
  /*
  DPCT1003:32: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c_tmp->R, c->R,
                              sizeof(float) * num_dimensions * num_dimensions *
                                  num_clusters)
                      .wait(),
                  0));
  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((q_ct1
                      .memcpy(c_tmp->Rinv, c->Rinv,
                              sizeof(float) * num_dimensions * num_dimensions *
                                  num_clusters)
                      .wait(),
                  0));
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

clusters_t *cluster(int original_num_clusters, int desired_num_clusters,
                    int *final_num_clusters, int num_dimensions, int num_events,
                    float *fcs_data_by_event) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

  int regroup_iterations = 0;
  int params_iterations = 0;
  int reduce_iterations = 0;
  int ideal_num_clusters = original_num_clusters;
  int stop_number;

  // Number of clusters to stop iterating at.
  if(desired_num_clusters == 0) {
    stop_number = 1;
  } else {
    stop_number = desired_num_clusters;
  }

  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      if(isnan(fcs_data_by_event[e*num_dimensions+d])) {
        printf("Error: Found NaN value in input data. Exiting.\n");
        return NULL;
      }
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }    


  PRINT("Number of events: %d\n",num_events);
  PRINT("Number of dimensions: %d\n\n",num_dimensions);
  PRINT("Starting with %d cluster(s), will stop at %d cluster(s).\n",original_num_clusters,stop_number);

  // This the shared memory space between the GPUs
  clusters_t clusters;
  setupCluster(&clusters, original_num_clusters, num_events, num_dimensions);


  // another set of clusters for saving the results of the best configuration
  clusters_t *saved_clusters = (clusters_t*) malloc(sizeof(clusters_t));
  setupCluster(saved_clusters, original_num_clusters, num_events, num_dimensions);

  DEBUG("Finished allocating shared cluster structures on host\n");

  // hold the result from regroup kernel
  float* shared_likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS);
  float likelihood, old_likelihood;
  float min_rissanen = FLT_MAX;

  // Used as a temporary cluster for combining clusters in "distance" computations
  clusters_t scratch_cluster;
  setupCluster(&scratch_cluster, 1, num_events, num_dimensions);

  DEBUG("Finished allocating memory on host for clusters.\n");

  // Setup the cluster data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  clusters_t temp_clusters;
  clusters_t *d_clusters = setupClusterDevice(&temp_clusters, original_num_clusters, num_events, num_dimensions);

  // allocate device memory for FCS data
  float* d_fcs_data_by_event;
  float* d_fcs_data_by_dimension;

  // allocate and copy relavant FCS data to device.
  int mem_size = num_dimensions * num_events * sizeof(float);
  /*
  DPCT1003:34: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (d_fcs_data_by_event = (float *)sycl::malloc_device(mem_size, q_ct1), 0));
  /*
  DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (d_fcs_data_by_dimension = (float *)sycl::malloc_device(mem_size, q_ct1),
       0));
  /*
  DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(d_fcs_data_by_event, fcs_data_by_event, mem_size).wait(),
       0));
  /*
  DPCT1003:37: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (q_ct1.memcpy(d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size)
           .wait(),
       0));

  DEBUG("GPU: Finished copying FCS data to device.\n");

  //////////////// Initialization done, starting kernels //////////////// 
  DEBUG("Invoking seed_clusters kernel.\n");

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        means_acc_ct1(sycl::range<1>(24 /*NUM_DIMENSIONS*/), cgh);
    sycl::accessor<float, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        avgvar_acc_ct1(cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        variances_acc_ct1(sycl::range<1>(24 /*NUM_DIMENSIONS*/), cgh);
    sycl::accessor<float, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        total_variance_acc_ct1(cgh);

    cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_THREADS_MSTEP),
                                       sycl::range<3>(1, 1, NUM_THREADS_MSTEP)),
                     [=](sycl::nd_item<3> item_ct1) {
                       seed_clusters_kernel(
                           d_fcs_data_by_event, d_clusters, num_dimensions,
                           original_num_clusters, num_events, item_ct1,
                           means_acc_ct1.get_pointer(),
                           avgvar_acc_ct1.get_pointer(),
                           variances_acc_ct1.get_pointer(),
                           total_variance_acc_ct1.get_pointer());
                     });
  });

  // Computes the R matrix inverses, and the gaussian constant
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        determinant_arg_acc_ct1(cgh);
    sycl::accessor<float, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        sum_acc_ct1(cgh);
    sycl::accessor<float, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        matrix_acc_ct1(sycl::range<1>(576 /*NUM_DIMENSIONS*NUM_DIMENSIONS*/),
                       cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, original_num_clusters) *
                              sycl::range<3>(1, 1, NUM_THREADS_MSTEP),
                          sycl::range<3>(1, 1, NUM_THREADS_MSTEP)),
        [=](sycl::nd_item<3> item_ct1) {
          constants_kernel(d_clusters, original_num_clusters, num_dimensions,
                           item_ct1, determinant_arg_acc_ct1.get_pointer(),
                           sum_acc_ct1.get_pointer(),
                           matrix_acc_ct1.get_pointer());
        });
  });

  // copy clusters from the device
  copyClusterFromDevice(&clusters, &temp_clusters, d_clusters, original_num_clusters, num_dimensions);

  //seed_clusters(&clusters,fcs_data_by_event,original_num_clusters,num_dimensions,num_events);

  DEBUG("Starting Clusters\n");
  for(int c=0; c < original_num_clusters; c++) {
    DEBUG("Cluster #%d\n",c);

    DEBUG("\tN: %f\n",clusters.N[c]); 
    DEBUG("\tpi: %f\n",clusters.pi[c]); 

    // means
    DEBUG("\tMeans: ");
    for(int d=0; d < num_dimensions; d++) {
      DEBUG("%.2f ",clusters.means[c*num_dimensions+d]);
    }
    DEBUG("\n");

    DEBUG("\tR:\n\t");
    for(int d=0; d < num_dimensions; d++) {
      for(int e=0; e < num_dimensions; e++)
        DEBUG("%.2f ",clusters.R[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
      DEBUG("\n\t");
    }
    DEBUG("R-inverse:\n\t");
    for(int d=0; d < num_dimensions; d++) {
      for(int e=0; e < num_dimensions; e++)
        DEBUG("%.2f ",clusters.Rinv[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
      DEBUG("\n\t");
    }
    DEBUG("\n");
    DEBUG("\tAvgvar: %e\n",clusters.avgvar[c]);
    DEBUG("\tConstant: %e\n",clusters.constant[c]);

  }

  // synchronize after first gpu does the seeding, copy result to all gpus
  copyClusterToDevice(&clusters, &temp_clusters, original_num_clusters, num_dimensions);

  // Calculate an epsilon value
  float epsilon = (1+num_dimensions+0.5f*(num_dimensions+1)*num_dimensions)*
    ::logf((float)num_events*num_dimensions)*0.001f;
  int iters;

  //epsilon = 1e-6;
  PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

  float* d_likelihoods;
  /*
  DPCT1003:38: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL(
      (d_likelihoods = sycl::malloc_device<float>(NUM_BLOCKS, q_ct1), 0));

  // Variables for GMM reduce order
  float distance, min_distance = 0.0;
  float rissanen;
  int min_c1, min_c2;

  for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
    /*************** EM ALGORITHM *****************************/

    // do initial E-step
    // Calculates a cluster membership probability
    // for each event and each cluster.
    DEBUG("Invoking E-step kernels.");
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          means_acc_ct1(sycl::range<1>(24 /*NUM_DIMENSIONS*/), cgh);
      sycl::accessor<float, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          Rinv_acc_ct1(sycl::range<1>(576 /*NUM_DIMENSIONS*NUM_DIMENSIONS*/),
                       cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, NUM_BLOCKS, num_clusters) *
                                sycl::range<3>(1, 1, NUM_THREADS_ESTEP),
                            sycl::range<3>(1, 1, NUM_THREADS_ESTEP)),
          [=](sycl::nd_item<3> item_ct1) {
            estep1(d_fcs_data_by_dimension, d_clusters, num_dimensions,
                   num_events, item_ct1, means_acc_ct1.get_pointer(),
                   Rinv_acc_ct1.get_pointer());
          });
    });
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<float, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          total_likelihoods_acc_ct1(sycl::range<1>(256 /*NUM_THREADS_ESTEP*/),
                                    cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                                sycl::range<3>(1, 1, NUM_THREADS_ESTEP),
                            sycl::range<3>(1, 1, NUM_THREADS_ESTEP)),
          [=](sycl::nd_item<3> item_ct1) {
            estep2(d_fcs_data_by_dimension, d_clusters, num_dimensions,
                   num_clusters, num_events, d_likelihoods, item_ct1,
                   total_likelihoods_acc_ct1.get_pointer());
          });
    });
    regroup_iterations++;

    // Copy the likelihood totals from each block, sum them up to get a total
    /*
    DPCT1003:39: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((q_ct1
                        .memcpy(shared_likelihoods, d_likelihoods,
                                sizeof(float) * NUM_BLOCKS)
                        .wait(),
                    0));
    likelihood = 0.0;
    for(int i=0;i<NUM_BLOCKS;i++) {
      likelihood += shared_likelihoods[i]; 
    }
    DEBUG("Likelihood: %e\n",likelihood);

    float change = epsilon*2;

    PRINT("Performing EM algorithm on %d clusters.\n",num_clusters);
    iters = 0;
    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < MIN_ITERS || (fabs(change) > epsilon && iters < MAX_ITERS)) {
      old_likelihood = likelihood;

      DEBUG("Invoking reestimate_parameters (M-step) kernel.");

      // This kernel computes a new N, pi isn't updated until compute_constants though
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            temp_sums_acc_ct1(sycl::range<1>(256 /*NUM_THREADS_MSTEP*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_clusters) *
                                  sycl::range<3>(1, 1, NUM_THREADS_MSTEP),
                              sycl::range<3>(1, 1, NUM_THREADS_MSTEP)),
            [=](sycl::nd_item<3> item_ct1) {
              mstep_N(d_clusters, num_dimensions, num_clusters, num_events,
                      item_ct1, temp_sums_acc_ct1.get_pointer());
            });
      });

      /*
      DPCT1003:42: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((
          q_ct1
              .memcpy(clusters.N, temp_clusters.N, sizeof(float) * num_clusters)
              .wait(),
          0));

      sycl::range<3> gridDim1(1, num_dimensions, num_clusters);
      sycl::range<3> blockDim1(1, 1, NUM_THREADS_MSTEP);
      /*
      DPCT1049:40: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            temp_sum_acc_ct1(sycl::range<1>(256 /*NUM_THREADS_MSTEP*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(gridDim1 * blockDim1, blockDim1),
                         [=](sycl::nd_item<3> item_ct1) {
                           mstep_means(d_fcs_data_by_dimension, d_clusters,
                                       num_dimensions, num_clusters, num_events,
                                       item_ct1,
                                       temp_sum_acc_ct1.get_pointer());
                         });
      });
      /*
      DPCT1003:43: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(clusters.means, temp_clusters.means,
                                  sizeof(float) * num_clusters * num_dimensions)
                          .wait(),
                      0));

      // Reduce means for all clusters, copy back to device
      for(int c=0; c < num_clusters; c++) {
        DEBUG("Cluster %d  Means:", c);
        for(int d=0; d < num_dimensions; d++) {
          if(clusters.N[c] > 0.5f) {
            clusters.means[c*num_dimensions+d] /= clusters.N[c];
          } else {
            clusters.means[c*num_dimensions+d] = 0.0f;
          }
          DEBUG(" %f",clusters.means[c*num_dimensions+d]);
        }
        DEBUG("\n");
      }
      /*
      DPCT1003:44: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(temp_clusters.means, clusters.means,
                                  sizeof(float) * num_clusters * num_dimensions)
                          .wait(),
                      0));

      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      sycl::range<3> gridDim2(1, num_dimensions * (num_dimensions + 1) / 2,
                              (num_clusters + NUM_CLUSTERS_PER_BLOCK - 1) /
                                  NUM_CLUSTERS_PER_BLOCK);
      /*
      DPCT1049:41: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            means_row_acc_ct1(sycl::range<1>(6 /*NUM_CLUSTERS_PER_BLOCK*/),
                              cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            means_col_acc_ct1(sycl::range<1>(6 /*NUM_CLUSTERS_PER_BLOCK*/),
                              cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            temp_sums_acc_ct1(
                sycl::range<1>(
                    1536 /*NUM_THREADS_MSTEP*NUM_CLUSTERS_PER_BLOCK*/),
                cgh);

        cgh.parallel_for(sycl::nd_range<3>(gridDim2 * blockDim1, blockDim1),
                         [=](sycl::nd_item<3> item_ct1) {
                           mstep_covariance2(d_fcs_data_by_dimension,
                                             d_clusters, num_dimensions,
                                             num_clusters, num_events, item_ct1,
                                             means_row_acc_ct1.get_pointer(),
                                             means_col_acc_ct1.get_pointer(),
                                             temp_sums_acc_ct1.get_pointer());
                         });
      });
      /*
      DPCT1003:45: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(clusters.R, temp_clusters.R,
                                  sizeof(float) * num_clusters *
                                      num_dimensions * num_dimensions)
                          .wait(),
                      0));

    DEBUG("After cov2\tR:\n\t");
    for(int c=0; c < num_clusters; c++) 
     for(int d=0; d < num_dimensions; d++) 
       for(int e=0; e < num_dimensions; e++)
         DEBUG("%.2f ",clusters.R[c*num_dimensions*num_dimensions+d*num_dimensions+e]);
    DEBUG("\n");
    
      // Reduce R for all clusters, copy back to device
      {
        for(int c=0; c < num_clusters; c++) {
          if(clusters.N[c] > 0.5f) {
            for(int d=0; d < num_dimensions*num_dimensions; d++) {
              clusters.R[c*num_dimensions*num_dimensions+d] /= clusters.N[c];
            }
          } else {
            for(int i=0; i < num_dimensions; i++) {
              for(int j=0; j < num_dimensions; j++) {
                if(i == j) {
                  clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 1.0;
                } else {
                  clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j] = 0.0;
                }
              }
            }
          }
        }
      }
      /*
      DPCT1003:46: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(temp_clusters.R, clusters.R,
                                  sizeof(float) * num_clusters *
                                      num_dimensions * num_dimensions)
                          .wait(),
                      0));

      //CUT_CHECK_ERROR("M-step Kernel execution failed: ");
      params_iterations++;

      DEBUG("Invoking constants kernel.");

      // Inverts the R matrices, computes the constant, normalizes cluster probabilities
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            determinant_arg_acc_ct1(cgh);
        sycl::accessor<float, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            sum_acc_ct1(cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            matrix_acc_ct1(
                sycl::range<1>(576 /*NUM_DIMENSIONS*NUM_DIMENSIONS*/), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, num_clusters) *
                                  sycl::range<3>(1, 1, NUM_THREADS_MSTEP),
                              sycl::range<3>(1, 1, NUM_THREADS_MSTEP)),
            [=](sycl::nd_item<3> item_ct1) {
              constants_kernel(d_clusters, num_clusters, num_dimensions,
                               item_ct1, determinant_arg_acc_ct1.get_pointer(),
                               sum_acc_ct1.get_pointer(),
                               matrix_acc_ct1.get_pointer());
            });
      });
      /*
      DPCT1003:47: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(clusters.constant, temp_clusters.constant,
                                  sizeof(float) * num_clusters)
                          .wait(),
                      0));
      for(int temp_c=0; temp_c < num_clusters; temp_c++)
        DEBUG("Cluster %d constant: %e\n",temp_c,clusters.constant[temp_c]);

      DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n",NUM_BLOCKS);
      // Compute new cluster membership probabilities for all the events
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            means_acc_ct1(sycl::range<1>(24 /*NUM_DIMENSIONS*/), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            Rinv_acc_ct1(sycl::range<1>(576 /*NUM_DIMENSIONS*NUM_DIMENSIONS*/),
                         cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, NUM_BLOCKS, num_clusters) *
                                  sycl::range<3>(1, 1, NUM_THREADS_ESTEP),
                              sycl::range<3>(1, 1, NUM_THREADS_ESTEP)),
            [=](sycl::nd_item<3> item_ct1) {
              estep1(d_fcs_data_by_dimension, d_clusters, num_dimensions,
                     num_events, item_ct1, means_acc_ct1.get_pointer(),
                     Rinv_acc_ct1.get_pointer());
            });
      });
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            total_likelihoods_acc_ct1(sycl::range<1>(256 /*NUM_THREADS_ESTEP*/),
                                      cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                                  sycl::range<3>(1, 1, NUM_THREADS_ESTEP),
                              sycl::range<3>(1, 1, NUM_THREADS_ESTEP)),
            [=](sycl::nd_item<3> item_ct1) {
              estep2(d_fcs_data_by_dimension, d_clusters, num_dimensions,
                     num_clusters, num_events, d_likelihoods, item_ct1,
                     total_likelihoods_acc_ct1.get_pointer());
            });
      });
      regroup_iterations++;

      // check if kernel execution generated an error
      //CUT_CHECK_ERROR("Kernel execution failed");

      // Copy the likelihood totals from each block, sum them up to get a total
      /*
      DPCT1003:48: Migrated API does not return error code. (*, 0) is inserted.
      You may need to rewrite this code.
      */
      CUDA_SAFE_CALL((q_ct1
                          .memcpy(shared_likelihoods, d_likelihoods,
                                  sizeof(float) * NUM_BLOCKS)
                          .wait(),
                      0));
      {
        likelihood = 0.0;
        for(int i=0;i<NUM_BLOCKS;i++) {
          likelihood += shared_likelihoods[i]; 
        }
        DEBUG("Likelihood: %e\n",likelihood);
      }
      change = likelihood - old_likelihood;
      DEBUG("GPU 0: Change in likelihood: %e\n",change);

      iters++;
    }

    DEBUG("GPU done with EM loop\n");

    // copy all of the arrays from the device
    copyClusterFromDevice(&clusters, &temp_clusters, NULL, num_clusters, num_dimensions);

    /*
    DPCT1003:49: Migrated API does not return error code. (*, 0) is inserted.
    You may need to rewrite this code.
    */
    CUDA_SAFE_CALL((q_ct1
                        .memcpy(clusters.memberships, temp_clusters.memberships,
                                sizeof(float) * num_events * num_clusters)
                        .wait(),
                    0));

    DEBUG("GPU done with copying cluster data from device\n");

    // Calculate Rissanen Score
    rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+num_dimensions+0.5f*(num_dimensions+1.0f)*num_dimensions)-1.0f)*::logf((float)num_events*num_dimensions);
    PRINT("\nLikelihood: %e\n",likelihood);
    PRINT("\nRissanen Score: %e\n",rissanen);

    // Save the cluster data the first time through, so we have a base rissanen score and result
    // Save the cluster data if the solution is better and the user didn't specify a desired number
    // If the num_clusters equals the desired number, stop
    if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) || (num_clusters == desired_num_clusters)) {
      min_rissanen = rissanen;
      ideal_num_clusters = num_clusters;
      memcpy(saved_clusters->N,clusters.N,sizeof(float)*num_clusters);
      memcpy(saved_clusters->pi,clusters.pi,sizeof(float)*num_clusters);
      memcpy(saved_clusters->constant,clusters.constant,sizeof(float)*num_clusters);
      memcpy(saved_clusters->avgvar,clusters.avgvar,sizeof(float)*num_clusters);
      memcpy(saved_clusters->means,clusters.means,sizeof(float)*num_dimensions*num_clusters);
      memcpy(saved_clusters->R,clusters.R,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
      memcpy(saved_clusters->Rinv,clusters.Rinv,sizeof(float)*num_dimensions*num_dimensions*num_clusters);
      memcpy(saved_clusters->memberships,clusters.memberships,sizeof(float)*num_events*num_clusters);
    }
    /**************** Reduce GMM Order ********************/
    // Don't want to reduce order on the last iteration
    if(num_clusters > stop_number) {
      //startTimer(timers.cpu);
      {
        // First eliminate any "empty" clusters 
        for(int i=num_clusters-1; i >= 0; i--) {
          if(clusters.N[i] < 0.5) {
            DEBUG("Cluster #%d has less than 1 data point in it.\n",i);
            for(int j=i; j < num_clusters-1; j++) {
              copy_cluster(clusters,j,clusters,j+1,num_dimensions);
            }
            num_clusters--;
          }
        }

        min_c1 = 0;
        min_c2 = 1;
        DEBUG("Number of non-empty clusters: %d\n",num_clusters); 
        // For all combinations of subclasses...
        // If the number of clusters got really big might need to do a non-exhaustive search
        // Even with 100*99/2 combinations this doesn't seem to take too long
        for(int c1=0; c1<num_clusters;c1++) {
          for(int c2=c1+1; c2<num_clusters;c2++) {
            // compute distance function between the 2 clusters
            distance = cluster_distance(clusters,c1,c2,scratch_cluster,num_dimensions);

            // Keep track of minimum distance
            if((c1 ==0 && c2 == 1) || distance < min_distance) {
              min_distance = distance;
              min_c1 = c1;
              min_c2 = c2;
            }
          }
        }

        PRINT("\nMinimum distance between (%d,%d). Combining clusters\n",min_c1,min_c2);
        // Add the two clusters with min distance together
        add_clusters(clusters,min_c1,min_c2,scratch_cluster,num_dimensions);

        // Copy new combined cluster into the main group of clusters, compact them
        copy_cluster(clusters,min_c1,scratch_cluster,0,num_dimensions);

        for(int i=min_c2; i < num_clusters-1; i++) {
          //printf("Copying cluster %d to cluster %d\n",i+1,i);
          copy_cluster(clusters,i,clusters,i+1,num_dimensions);
        }
      }

      // Copy the clusters back to the device
      copyClusterToDevice(&clusters, &temp_clusters, num_clusters, num_dimensions);

    } // GMM reduction block 
    reduce_iterations++;

  } // outer loop from M to 1 clusters
  PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);

  /*
  DPCT1003:50: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(d_likelihoods, q_ct1), 0));
  /*
  DPCT1003:51: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(d_fcs_data_by_event, q_ct1), 0));
  /*
  DPCT1003:52: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(d_fcs_data_by_dimension, q_ct1), 0));
  /*
  DPCT1003:53: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  CUDA_SAFE_CALL((sycl::free(d_clusters, q_ct1), 0));

  freeCluster(&scratch_cluster);
  freeCluster(&clusters);
  freeClusterDevice(&temp_clusters);
  free(fcs_data_by_dimension);
  free(shared_likelihoods);

  *final_num_clusters = ideal_num_clusters;
  return saved_clusters;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
