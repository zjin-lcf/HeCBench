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

float cluster_distance(clusters_t &clusters, const int c1, const int c2, 
                       clusters_t &temp_cluster, const int num_dimensions) {
  // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
  add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);

  return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] 
         - temp_cluster.N[0]*temp_cluster.constant[0];
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


void copyClusterFromDevice(queue &q, 
                           clusters_t* c, 
                           buffer<float, 1> &d_N,
                           buffer<float, 1> &d_R,
                           buffer<float, 1> &d_Rinv,
                           buffer<float, 1> &d_pi,
                           buffer<float, 1> &d_constant,
                           buffer<float, 1> &d_avgvar,
                           buffer<float, 1> &d_means,
                           const int num_clusters, 
                           const int num_dimensions) {
  
  // CUDA_SAFE_CALL(cudaMemcpy(c->N, c_tmp->N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_N.get_access<sycl_read>(cgh, range<1>(num_clusters)); 
    cgh.copy(acc, c->N);
  });

  // CUDA_SAFE_CALL(cudaMemcpy(c->R, c_tmp->R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_R.get_access<sycl_read>(cgh, range<1>(num_dimensions*num_dimensions*num_clusters)); 
    cgh.copy(acc, c->R);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c->Rinv, c_tmp->Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_Rinv.get_access<sycl_read>(cgh, range<1>(num_dimensions*num_dimensions*num_clusters));
    cgh.copy(acc, c->Rinv);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c->pi, c_tmp->pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_pi.get_access<sycl_read>(cgh, range<1>(num_clusters));
    cgh.copy(acc, c->pi);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c->constant, c_tmp->constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_constant.get_access<sycl_read>(cgh, range<1>(num_clusters)); 
    cgh.copy(acc, c->constant);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c->avgvar, c_tmp->avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_avgvar.get_access<sycl_read>(cgh, range<1>(num_clusters)); 
    cgh.copy(acc, c->avgvar);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c->means, c_tmp->means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_means.get_access<sycl_read>(cgh, range<1>(num_dimensions*num_clusters)); 
    cgh.copy(acc, c->means);
  });
}

void copyClusterToDevice(queue &q, 
                         clusters_t* c, 
                         buffer<float, 1> &d_N,
                         buffer<float, 1> &d_R,
                         buffer<float, 1> &d_Rinv,
                         buffer<float, 1> &d_pi,
                         buffer<float, 1> &d_constant,
                         buffer<float, 1> &d_avgvar,
                         buffer<float, 1> &d_means,
                         const int num_clusters, 
                         const int num_dimensions) {
  
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->N, c->N, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_N.get_access<sycl_write>(cgh, range<1>(num_clusters)); 
    cgh.copy(c->N, acc);
  });

  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->R, c->R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_R.get_access<sycl_write>(cgh, range<1>(num_dimensions*num_dimensions*num_clusters)); 
    cgh.copy(c->R, acc);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->Rinv, c->Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_Rinv.get_access<sycl_write>(cgh, range<1>(num_dimensions*num_dimensions*num_clusters));
    cgh.copy(c->Rinv, acc);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->pi, c->pi, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_pi.get_access<sycl_write>(cgh, range<1>(num_clusters));
    cgh.copy(c->pi, acc);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->constant, c->constant, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_constant.get_access<sycl_write>(cgh, range<1>(num_clusters)); 
    cgh.copy(c->constant, acc);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->avgvar, c->avgvar, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_avgvar.get_access<sycl_write>(cgh, range<1>(num_clusters)); 
    cgh.copy(c->avgvar, acc);
  });
  // CUDA_SAFE_CALL(cudaMemcpy(c_tmp->means, c->means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
  q.submit([&] (handler &cgh) {
    auto acc = d_means.get_access<sycl_write>(cgh, range<1>(num_dimensions*num_clusters)); 
    cgh.copy(c->means, acc);
  });
}

clusters_t* cluster(int original_num_clusters, int desired_num_clusters, 
    int* final_num_clusters, int num_dimensions, int num_events, 
    float* fcs_data_by_event) {

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
  float *shared_likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS);
  float likelihood, old_likelihood;
  float min_rissanen = FLT_MAX;

  // Used as a temporary cluster for combining clusters in "distance" computations
  clusters_t scratch_cluster;
  setupCluster(&scratch_cluster, 1, num_events, num_dimensions);

  DEBUG("Finished allocating memory on host for clusters.\n");

  // Setup the cluster data structures on device
  // First allocate structures on the host, then allocate the arrays and copy them over
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // clusters_t *d_clusters = setupClusterDevice(&temp_clusters, original_num_clusters, num_events, num_dimensions);
  buffer<float, 1> d_N(original_num_clusters);
  buffer<float, 1> d_pi(original_num_clusters);
  buffer<float, 1> d_constant(original_num_clusters);
  buffer<float, 1> d_avgvar(original_num_clusters);
  buffer<float, 1> d_means(num_dimensions*original_num_clusters);
  buffer<float, 1> d_R(num_dimensions*num_dimensions*original_num_clusters);
  buffer<float, 1> d_Rinv(num_dimensions*num_dimensions*original_num_clusters);
  buffer<float, 1> d_memberships(num_events*(original_num_clusters+
                   NUM_CLUSTERS_PER_BLOCK-original_num_clusters % NUM_CLUSTERS_PER_BLOCK));


  // allocate and copy relavant FCS data to device.
  //CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  //CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
  //CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  buffer<float, 1> d_fcs_data_by_event (fcs_data_by_event, num_dimensions * num_events);
  buffer<float, 1> d_fcs_data_by_dimension (fcs_data_by_dimension, num_dimensions * num_events);

  DEBUG("GPU: Finished copying FCS data to device.\n");

  //////////////// Initialization done, starting kernels //////////////// 
  DEBUG("Invoking seed_clusters kernel.\n");

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  // seed_clusters_kernel<<< 1, NUM_THREADS_MSTEP >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
  q.submit([&] (handler &cgh) {
    auto fcs_data = d_fcs_data_by_event.get_access<sycl_read>(cgh);
    auto clusters_means = d_means.get_access<sycl_discard_write>(cgh);
    auto clusters_pi = d_pi.get_access<sycl_discard_write>(cgh);
    auto clusters_R = d_R.get_access<sycl_discard_write>(cgh);
    auto clusters_N = d_N.get_access<sycl_discard_write>(cgh);
    auto clusters_avgvar = d_avgvar.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> means(NUM_DIMENSIONS, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> avgvar(1, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> total_variance(1, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> variances(NUM_DIMENSIONS, cgh);
    cgh.parallel_for<class seed_clusters>(nd_range<1>(range<1>(NUM_THREADS_MSTEP), range<1>(NUM_THREADS_MSTEP)),
      [=] (nd_item<1> item) {
      int tid = item.get_local_id(0);
      int num_threads = item.get_local_range(0);
      float seed;
          
      // Number of elements in the covariance matrix
      int num_elements = num_dimensions*num_dimensions; 

      // Compute the means

      if(tid < num_dimensions) {
          means[tid] = 0.0;

          // Sum up all the values for each dimension
          for(int i = 0; i < num_events; i++) {
              means[tid] += fcs_data[i*num_dimensions+tid];
          }

          // Divide by the # of elements to get the average
          means[tid] /= (float) num_events;
      }

      item.barrier(access::fence_space::local_space);
      
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
      
      item.barrier(access::fence_space::local_space);
      
      if(tid == 0) {
        total_variance[0] = 0.0;
        for(int i=0; i<num_dimensions;i++)
          total_variance[0] += variances[i];
        avgvar[0] = total_variance[0] / (float) num_dimensions;
      }
          
      item.barrier(access::fence_space::local_space);

      if(original_num_clusters > 1) {
        seed = (num_events-1.0f)/(original_num_clusters-1.0f);
      } else {
        seed = 0.0;
      }
      
      // Seed the pi, means, and covariances for every cluster
      for(int c=0; c < original_num_clusters; c++) {
        if(tid < num_dimensions) {
            clusters_means[c*num_dimensions+tid] = fcs_data[((int)(c*seed))*num_dimensions+tid];
        }
          
        for(int i=tid; i < num_elements; i+= num_threads) {
            // Add the average variance divided by a constant, this keeps the cov matrix from becoming singular
            int row = (i) / num_dimensions;
            int col = (i) % num_dimensions;

            if(row == col) {
                clusters_R[c*num_dimensions*num_dimensions+i] = 1.0f;
            } else {
                clusters_R[c*num_dimensions*num_dimensions+i] = 0.0f;
            }
        }
        if(tid == 0) {
            clusters_pi[c] = 1.0f/((float)original_num_clusters);
            clusters_N[c] = ((float) num_events) / ((float)original_num_clusters);
            clusters_avgvar[c] = avgvar[0] / COVARIANCE_DYNAMIC_RANGE;
        }
      }
    });
  });

  // Computes the R matrix inverses, and the gaussian constant
  // constants_kernel<<<original_num_clusters, NUM_THREADS_MSTEP>>>(d_clusters,original_num_clusters,num_dimensions);
  range<1> constants_gws(original_num_clusters*NUM_THREADS_MSTEP);
  range<1> constants_lws(NUM_THREADS_MSTEP);
  q.submit([&] (handler &cgh) {
    auto clusters_R = d_R.get_access<sycl_read>(cgh);
    auto clusters_Rinv = d_Rinv.get_access<sycl_discard_write>(cgh);
    auto clusters_N = d_N.get_access<sycl_read>(cgh);
    auto clusters_pi = d_pi.get_access<sycl_discard_write>(cgh);
    auto clusters_constant = d_constant.get_access<sycl_discard_write>(cgh);
    auto clusters_avgvar = d_avgvar.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> matrix(NUM_DIMENSIONS*NUM_DIMENSIONS, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> determinant_arg(1, cgh);
    accessor<float, 1, sycl_read_write, access::target::local> sum(1, cgh);
    cgh.parallel_for<class constants>(nd_range<1>(constants_gws, constants_lws), [=] (nd_item<1> item) {
      constants_kernel( item, 
                        clusters_R.get_pointer(), 
                        clusters_Rinv.get_pointer(), 
                        clusters_N.get_pointer(), 
                        clusters_pi.get_pointer(), 
                        clusters_constant.get_pointer(), 
                        clusters_avgvar.get_pointer(), 
                        matrix.get_pointer(),
                        determinant_arg.get_pointer(),
                        sum.get_pointer(),
                        original_num_clusters, 
                        num_dimensions);
    });
  });

  // copy clusters from the device
  copyClusterFromDevice(q, &clusters, d_N, d_R, d_Rinv, d_pi, d_constant, d_avgvar, d_means, 
                        original_num_clusters, num_dimensions);
  q.wait();

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
  copyClusterToDevice(q, &clusters, d_N, d_R, d_Rinv, d_pi, d_constant, d_avgvar, d_means,
                          original_num_clusters, num_dimensions);

  // Calculate an epsilon value
  float epsilon = (1+num_dimensions+0.5f*(num_dimensions+1)*num_dimensions)*
    ::logf((float)num_events*num_dimensions)*0.001f;
  int iters;

  //epsilon = 1e-6;
  PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

  //float* d_likelihoods;
  //CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS));
  buffer<float, 1> d_likelihoods (NUM_BLOCKS);

  // Variables for GMM reduce order
  float distance, min_distance = 0.0;
  float rissanen;
  int min_c1, min_c2;

  for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {

    // *************** EM ALGORITHM *****************************
    // do initial E-step
    // Calculates a cluster membership probability
    // for each event and each cluster.
    DEBUG("Invoking E-step kernels.");
  //  estep1<<<dim3(num_clusters,NUM_BLOCKS), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events);
    range<2> estep1_gws (NUM_BLOCKS, num_clusters*NUM_THREADS_ESTEP);
    range<2> estep1_lws (1, NUM_THREADS_ESTEP);
    q.submit([&] (handler &cgh) {
      auto data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_Rinv = d_Rinv.get_access<sycl_read>(cgh);
      auto clusters_pi = d_pi.get_access<sycl_read>(cgh);
      auto clusters_constant = d_constant.get_access<sycl_read>(cgh);
      auto clusters_means = d_means.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> means (NUM_DIMENSIONS, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> Rinv (NUM_DIMENSIONS*NUM_DIMENSIONS, cgh);
      cgh.parallel_for<class estep1>(nd_range<2>(estep1_gws, estep1_lws), [=] (nd_item<2> item) {
        estep1_kernel( 
            item,
            data.get_pointer(), 
            clusters_Rinv.get_pointer(), 
            clusters_memberships.get_pointer(), 
            clusters_pi.get_pointer(), 
            clusters_constant.get_pointer(), 
            clusters_means.get_pointer(), 
            means.get_pointer(), 
            Rinv.get_pointer(), 
            num_dimensions, 
            num_events); 
      });
    });

    //  estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
    range<1> estep2_gws (NUM_BLOCKS*NUM_THREADS_ESTEP);
    range<1> estep2_lws (NUM_THREADS_ESTEP);
    q.submit([&] (handler &cgh) {
      //auto fcs_data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_read_write>(cgh);
      auto likelihood = d_likelihoods.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> total_likelihoods (NUM_THREADS_ESTEP, cgh);
      cgh.parallel_for<class estep2>(nd_range<1>(estep2_gws, estep2_lws), [=] (nd_item<1> item) {
        estep2_kernel(
            item,
            clusters_memberships.get_pointer(), 
            likelihood.get_pointer(), 
            total_likelihoods.get_pointer(), 
            num_dimensions, 
            num_clusters, 
            num_events); 
       });
    });

    regroup_iterations++;

    // Copy the likelihood totals from each block, sum them up to get a total
    // CUDA_SAFE_CALL(cudaMemcpy(shared_likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
    q.submit([&] (handler &cgh) {
      auto acc = d_likelihoods.get_access<sycl_read>(cgh); 
      cgh.copy(acc, shared_likelihoods);
    });
    q.wait();

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
      //mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(d_clusters,num_dimensions,num_clusters,num_events);
    range<1> mStepN_gws (num_clusters*NUM_THREADS_MSTEP);
    range<1> mStepN_lws (NUM_THREADS_MSTEP);
    q.submit([&] (handler &cgh) {
      // auto fcs_data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_read>(cgh);
      auto clusters_N = d_N.get_access<sycl_discard_write>(cgh);
      auto clusters_pi = d_pi.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> temp_sums (NUM_THREADS_MSTEP, cgh);
      cgh.parallel_for<class estep2>(nd_range<1>(mStepN_gws, mStepN_lws), [=] (nd_item<1> item) {
        int tid = item.get_local_id(0);
        int num_threads = item.get_local_range(0);
        int c = item.get_group(0);
 
        // Compute new N
        float sum = 0.0f;
        // Break all the events accross the threads, add up probabilities
        for(int event=tid; event < num_events; event += num_threads) {
            sum += clusters_memberships[c*num_events+event];
        }
        temp_sums[tid] = sum;
 
        item.barrier(access::fence_space::local_space);

        // sum = parallelSum(temp_sums,NUM_THREADS_MSTEP);

        for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
          float t = temp_sums[tid] + temp_sums[tid^bit];
          item.barrier(access::fence_space::local_space);
          temp_sums[tid] = t;
          item.barrier(access::fence_space::local_space);
        }
        sum = temp_sums[tid];
        if(tid == 0) {
          clusters_N[c] = sum;
          clusters_pi[c] = sum;
        }
      });
    });
 
      //CUDA_SAFE_CALL(cudaMemcpy(clusters.N,temp_clusters.N,sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    q.submit([&] (handler &cgh) {
      auto acc = d_N.get_access<sycl_read>(cgh, range<1>(num_clusters)); 
      cgh.copy(acc, clusters.N);
    });


    //  mstep_means<<<gridDim1, blockDim1>>>(d_fcs_data_by_dimension,d_clusters, num_dimensions,num_clusters,num_events);
    range<2> mstepMeans_gws (num_dimensions, num_clusters*NUM_THREADS_MSTEP);
    range<2> mstepMeans_lws (1, NUM_THREADS_MSTEP);

    q.submit([&] (handler &cgh) {
      auto fcs_data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_read>(cgh);
      auto clusters_means = d_means.get_access<sycl_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> temp_sum (NUM_THREADS_MSTEP, cgh);
      cgh.parallel_for<class mstepMeans>(nd_range<2>(mstepMeans_gws, mstepMeans_lws), [=] (nd_item<2> item) {
        int tid = item.get_local_id(1);
        int num_threads = item.get_local_range(1);
        int c = item.get_group(1);
        int d = item.get_group(0); // dimension number

        float sum = 0.0f;
        for(int event=tid; event < num_events; event+= num_threads) {
            sum += fcs_data[d*num_events+event]*clusters_memberships[c*num_events+event];
        }
        temp_sum[tid] = sum;
        
        item.barrier(access::fence_space::local_space);

        // Reduce partial sums
        // sum = parallelSum(temp_sum,NUM_THREADS_MSTEP);
        for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
          float t = temp_sum[tid] + temp_sum[tid^bit];
          item.barrier(access::fence_space::local_space);
          temp_sum[tid] = t;
          item.barrier(access::fence_space::local_space);
        }
        sum = temp_sum[tid];
        if(tid == 0) clusters_means[c*num_dimensions+d] = sum;
     });
   });


    //  CUDA_SAFE_CALL(cudaMemcpy(clusters.means,temp_clusters.means,
      //      sizeof(float)*num_clusters*num_dimensions,cudaMemcpyDeviceToHost));
    q.submit([&] (handler &cgh) {
      auto acc = d_means.get_access<sycl_read>(cgh, range<1>(num_clusters*num_dimensions)); 
      cgh.copy(acc, clusters.means);
    });

    q.wait();

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
    //  CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means,clusters.means,
     //       sizeof(float)*num_clusters*num_dimensions,cudaMemcpyHostToDevice));
    q.submit([&] (handler &cgh) {
      auto acc = d_means.get_access<sycl_write>(cgh, range<1>(num_clusters*num_dimensions)); 
      cgh.copy(clusters.means, acc);
    });


      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      //dim3 gridDim2((num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK, num_dimensions*(num_dimensions+1)/2);
      // mstep_covariance2<<<gridDim2, blockDim1>>>(d_fcs_data_by_dimension,d_clusters,
         // num_dimensions,num_clusters,num_events);
      range<2> cov2_gws(num_dimensions*(num_dimensions+1)/2, 
                        (num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK*NUM_THREADS_MSTEP); 
      range<2> cov2_lws (1, NUM_THREADS_MSTEP);

    q.submit([&] (handler &cgh) {
      auto fcs_data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_write>(cgh);
      auto clusters_means = d_means.get_access<sycl_read>(cgh);
      auto clusters_R = d_R.get_access<sycl_read_write>(cgh);
      auto clusters_avgvar = d_avgvar.get_access<sycl_read>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> means_row (NUM_CLUSTERS_PER_BLOCK, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> means_col (NUM_CLUSTERS_PER_BLOCK, cgh);
      // 256 * 6
      accessor<float, 1, sycl_read_write, access::target::local> temp_sums (NUM_THREADS_MSTEP*NUM_CLUSTERS_PER_BLOCK, cgh);
      cgh.parallel_for<class covariance2>(nd_range<2>(cov2_gws, cov2_lws), [=] (nd_item<2> item) {
        int tid = item.get_local_id(1);

        // Determine what row,col this matrix is handling, also handles the symmetric element
        int row,col,c1;
        compute_row_col(item, num_dimensions, &row, &col);

        item.barrier(access::fence_space::local_space);
        
         c1 = item.get_group(1) * NUM_CLUSTERS_PER_BLOCK; // Determines what cluster this block is handling    

        #if DIAG_ONLY
        if(row != col) {
            clusters_R[c*num_dimensions*num_dimensions+row*num_dimensions+col] = 0.0f;
            clusters_R[c*num_dimensions*num_dimensions+col*num_dimensions+row] = 0.0f;
            return;
        }
        #endif 

        if ( (tid < min(num_clusters, NUM_CLUSTERS_PER_BLOCK))  // c1 = 0
             && (c1+tid < num_clusters)) { 
            means_row[tid] = clusters_means[(c1+tid)*num_dimensions+row];
            means_col[tid] = clusters_means[(c1+tid)*num_dimensions+col];
        }

        // Sync to wait for all params to be loaded to shared memory
        item.barrier(access::fence_space::local_space);

        
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
            cov_sum1 += (val1-means_row[0])*(val2-means_col[0])*clusters_memberships[c1*num_events+event]; 
            cov_sum2 += (val1-means_row[1])*(val2-means_col[1])*clusters_memberships[(c1+1)*num_events+event]; 
            cov_sum3 += (val1-means_row[2])*(val2-means_col[2])*clusters_memberships[(c1+2)*num_events+event]; 
            cov_sum4 += (val1-means_row[3])*(val2-means_col[3])*clusters_memberships[(c1+3)*num_events+event]; 
            cov_sum5 += (val1-means_row[4])*(val2-means_col[4])*clusters_memberships[(c1+4)*num_events+event]; 
            cov_sum6 += (val1-means_row[5])*(val2-means_col[5])*clusters_memberships[(c1+5)*num_events+event]; 
        }
        temp_sums[0*NUM_THREADS_MSTEP+tid] = cov_sum1;
        temp_sums[1*NUM_THREADS_MSTEP+tid] = cov_sum2;
        temp_sums[2*NUM_THREADS_MSTEP+tid] = cov_sum3;
        temp_sums[3*NUM_THREADS_MSTEP+tid] = cov_sum4;
        temp_sums[4*NUM_THREADS_MSTEP+tid] = cov_sum5;
        temp_sums[5*NUM_THREADS_MSTEP+tid] = cov_sum6;

        item.barrier(access::fence_space::local_space);
   
        for (int c=0; c < NUM_CLUSTERS_PER_BLOCK; c++) {
          //temp_sums[c*NUM_THREADS_MSTEP+tid] = parallelSum(&temp_sums[c*NUM_THREADS_MSTEP],NUM_THREADS_MSTEP);
          float *temp_sum = &temp_sums[c*NUM_THREADS_MSTEP];
          for(unsigned int bit = NUM_THREADS_MSTEP >> 1; bit > 0; bit >>= 1) {
            float t = temp_sum[tid] + temp_sum[tid^bit];
            item.barrier(access::fence_space::local_space);
            temp_sum[tid] = t;
            item.barrier(access::fence_space::local_space);
          }
          temp_sums[c*NUM_THREADS_MSTEP+tid] = temp_sum[tid];
          item.barrier(access::fence_space::local_space);
        }
        
        if (tid == 0) {
          for (int c=0; c < NUM_CLUSTERS_PER_BLOCK && (c+c1) < num_clusters; c++) {
            int offset = (c+c1)*num_dimensions*num_dimensions;
            cov_sum1 = temp_sums[c*NUM_THREADS_MSTEP];
            clusters_R[offset+row*num_dimensions+col] = cov_sum1;
            // Set the symmetric value
            clusters_R[offset+col*num_dimensions+row] = cov_sum1;
            
            // Regularize matrix - adds some variance to the diagonal elements
            // Helps keep covariance matrix non-singular (so it can be inverted)
            // The amount added is scaled down based on COVARIANCE_DYNAMIC_RANGE constant defined in gaussian.h
            if(row == col) clusters_R[offset+row*num_dimensions+col] += clusters_avgvar[c+c1];
          }
        }
      });
    });

    //  CUDA_SAFE_CALL(cudaMemcpy(clusters.R,temp_clusters.R,
     //       sizeof(float)*num_clusters*num_dimensions*num_dimensions,cudaMemcpyDeviceToHost));
    q.submit([&] (handler &cgh) {
      auto acc = d_R.get_access<sycl_read>(cgh, range<1>(num_clusters*num_dimensions*num_dimensions)); 
      cgh.copy(acc, clusters.R);
    });

    q.wait();

    // Reduce R for all clusters, copy back to device
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

    // CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R,clusters.R,
    //     sizeof(float)*num_clusters*num_dimensions*num_dimensions,cudaMemcpyHostToDevice));
    q.submit([&] (handler &cgh) {
      auto acc = d_R.get_access<sycl_write>(cgh, range<1>(num_clusters*num_dimensions*num_dimensions)); 
      cgh.copy(clusters.R, acc);
    });


    //CUT_CHECK_ERROR("M-step Kernel execution failed: ");
    params_iterations++;

    DEBUG("Invoking constants kernel.");

    // Inverts the R matrices, computes the constant, normalizes cluster probabilities
    // constants_kernel<<<num_clusters, NUM_THREADS_MSTEP>>>(d_clusters,num_clusters,num_dimensions);
    range<1> constants2_gws(num_clusters*NUM_THREADS_MSTEP);
    q.submit([&] (handler &cgh) {
      auto clusters_R = d_R.get_access<sycl_read>(cgh);
      auto clusters_Rinv = d_Rinv.get_access<sycl_discard_write>(cgh);
      auto clusters_N = d_N.get_access<sycl_read>(cgh);
      auto clusters_pi = d_pi.get_access<sycl_discard_write>(cgh);
      auto clusters_constant = d_constant.get_access<sycl_discard_write>(cgh);
      auto clusters_avgvar = d_avgvar.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> matrix(NUM_DIMENSIONS*NUM_DIMENSIONS, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> determinant_arg(1, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> sum(1, cgh);
      cgh.parallel_for<class constants>(nd_range<1>(constants2_gws, constants_lws), [=] (nd_item<1> item) {
        constants_kernel( item, 
                          clusters_R.get_pointer(), 
                          clusters_Rinv.get_pointer(), 
                          clusters_N.get_pointer(), 
                          clusters_pi.get_pointer(), 
                          clusters_constant.get_pointer(), 
                          clusters_avgvar.get_pointer(), 
                          matrix.get_pointer(),
                          determinant_arg.get_pointer(),
                          sum.get_pointer(),
                          num_clusters, // original_num_clusters, 
                          num_dimensions);
      });
    });

    //    CUDA_SAFE_CALL(cudaMemcpy(clusters.constant, temp_clusters.constant, 
     //         sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
        
    q.submit([&] (handler &cgh) {
      auto acc = d_constant.get_access<sycl_read>(cgh, range<1>(num_clusters)); 
      cgh.copy(acc, clusters.constant);
    });
    q.wait();

    for(int temp_c=0; temp_c < num_clusters; temp_c++)
      DEBUG("Cluster %d constant: %e\n",temp_c,clusters.constant[temp_c]);

    DEBUG("Invoking regroup (E-step) kernel with %d blocks.\n",NUM_BLOCKS);

      // Compute new cluster membership probabilities for all the events
      // estep1<<<dim3(num_clusters,NUM_BLOCKS), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events);
    q.submit([&] (handler &cgh) {
      auto data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_Rinv = d_Rinv.get_access<sycl_read>(cgh);
      auto clusters_memberships = d_memberships.get_access<sycl_discard_write>(cgh);
      auto clusters_pi = d_pi.get_access<sycl_discard_write>(cgh);
      auto clusters_constant = d_constant.get_access<sycl_discard_write>(cgh);
      auto clusters_means = d_means.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> means (NUM_DIMENSIONS, cgh);
      accessor<float, 1, sycl_read_write, access::target::local> Rinv (NUM_DIMENSIONS*NUM_DIMENSIONS, cgh);
      cgh.parallel_for<class estep1>(nd_range<2>(estep1_gws, estep1_lws), [=] (nd_item<2> item) {
        estep1_kernel( 
            item,
            data.get_pointer(), 
            clusters_Rinv.get_pointer(), 
            clusters_memberships.get_pointer(), 
            clusters_pi.get_pointer(), 
            clusters_constant.get_pointer(), 
            clusters_means.get_pointer(), 
            means.get_pointer(),
            Rinv.get_pointer(),
            num_dimensions, 
            num_events); 
      });
    });
  //    estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
    q.submit([&] (handler &cgh) {
      // auto fcs_data = d_fcs_data_by_dimension.get_access<sycl_read>(cgh);
      auto clusters_memberships= d_memberships.get_access<sycl_discard_write>(cgh);
      auto likelihood = d_likelihoods.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> total_likelihoods (NUM_THREADS_ESTEP, cgh);
      cgh.parallel_for<class estep2>(nd_range<1>(estep2_gws, estep2_lws), [=] (nd_item<1> item) {
        estep2_kernel(
            item,
            clusters_memberships.get_pointer(), 
            likelihood.get_pointer(), 
            total_likelihoods.get_pointer(), 
            num_dimensions, 
            num_clusters, 
            num_events); 
      });
    });

    regroup_iterations++;

      // check if kernel execution generated an error
      //CUT_CHECK_ERROR("Kernel execution failed");

      // Copy the likelihood totals from each block, sum them up to get a total
  //    CUDA_SAFE_CALL(cudaMemcpy(shared_likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
    q.submit([&] (handler &cgh) {
      auto acc = d_likelihoods.get_access<sycl_read>(cgh); 
      cgh.copy(acc, shared_likelihoods);
    });
    q.wait();
    likelihood = 0.0;
    for(int i=0;i<NUM_BLOCKS;i++) likelihood += shared_likelihoods[i]; 
    DEBUG("Likelihood: %e\n",likelihood);
    change = likelihood - old_likelihood;
    DEBUG("GPU 0: Change in likelihood: %e\n",change);
    iters++;
  }

  DEBUG("GPU done with EM loop\n");

  // copy all of the arrays from the device
  copyClusterFromDevice(q, &clusters, d_N, d_R, d_Rinv, d_pi, d_constant, d_avgvar, d_means, 
                          num_clusters, num_dimensions);

    //  CUDA_SAFE_CALL(cudaMemcpy(clusters.memberships, temp_clusters.memberships, sizeof(float)*num_events*num_clusters,cudaMemcpyDeviceToHost));
  q.submit([&] (handler &cgh) {
    auto acc = d_memberships.get_access<sycl_read>(cgh, range<1>(num_events*num_clusters)); 
    cgh.copy(acc, clusters.memberships);
  });


  DEBUG("GPU done with copying cluster data from device\n");

  // Calculate Rissanen Score
  rissanen = -likelihood + 0.5f*(num_clusters*(1.0f+num_dimensions+0.5f*(num_dimensions+1.0f)*num_dimensions)-1.0f)*::logf((float)num_events*num_dimensions);
  PRINT("\nLikelihood: %e\n",likelihood);
  PRINT("\nRissanen Score: %e\n",rissanen);

  q.wait();
    // Save the cluster data the first time through, so we have a base rissanen score and result
    // Save the cluster data if the solution is better and the user didn't specify a desired number
    // If the num_clusters equals the desired number, stop
  if(num_clusters == original_num_clusters || (rissanen < min_rissanen && desired_num_clusters == 0) 
     || (num_clusters == desired_num_clusters)) {
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

      // Copy the clusters back to the device
      copyClusterToDevice(q, &clusters, d_N, d_R, d_Rinv, d_pi, d_constant, d_avgvar, d_means,
                          num_clusters, num_dimensions);

    } // GMM reduction block 
    reduce_iterations++;

  } // outer loop from M to 1 clusters
  PRINT("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,ideal_num_clusters);

  freeCluster(&scratch_cluster);
  freeCluster(&clusters);
  free(fcs_data_by_dimension);
  free(shared_likelihoods);

  *final_num_clusters = ideal_num_clusters;
  return saved_clusters;
}
