/**   Edited by: Shuai Che, David Tarjan, Sang-Ha Lee					**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					No longer performs "validity" function to analyze	**/
/**					compactness and separation crietria; instead		**/
/**					calculate root mean squared error.					**/
/**                                                                     **/
/*************************************************************************/

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
#include "kmeans.h"

float	min_rmse_ref = FLT_MAX;		
extern double wtime(void);
/* reference min_rmse value */

// copy the feature to a feature swap region
void feature_transpose (float* feature_swap, const float* feature, const int nfeatures, const int npoints,
                        sycl::nd_item<3> item_ct1)
{
  int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
  if (tid < npoints) {
    for(int i = 0; i <  nfeatures; i++)
      feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
  }
}

void find_membership (const float* feature, const float* cluster, int* member, 
    const int nclusters, const int nfeatures, const int npoints,
    sycl::nd_item<3> item_ct1)
{
  int point_id = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
  if (point_id < npoints) {
    int index = 0;
    float min_dist = FLT_MAX;
    for (int i = 0; i < nclusters; i++) {
      float dist = 0;
      float ans  = 0;
      for (int l = 0; l < nfeatures; l++) {
        ans += (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) * 
               (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) ; 
      }
      dist = ans;
      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
    member[point_id] = index;
  }
}


/*---< cluster() >-----------------------------------------------------------*/
int cluster(int      npoints,         /* number of data points */
            int      nfeatures,       /* number of attributes for each point */
            float  **features,        /* array: [npoints][nfeatures] */                  
            int      min_nclusters,   /* range of min to max number of clusters */
            int		 max_nclusters,
            float    threshold,       /* loop terminating factor */
            int     *best_nclusters,  /* out: number between min and max with lowest RMSE */
            float ***cluster_centres, /* out: [best_nclusters][nfeatures] */
            float	*min_rmse,          /* out: minimum RMSE */
            int		 isRMSE,            /* calculate RMSE */
            int		 nloops             /* number of iteration for each number of clusters */
    )
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int		index =0;	/* number of iteration to reach the best RMSE */
  int		rmse;     /* RMSE for each clustering */
  float delta;

  /* current memberships of points  */
  int *membership = (int*) malloc(npoints * sizeof(int));

  /* new memberships of points computed by the device */
  int *membership_OCL = (int*) malloc(npoints * sizeof(int));

  // associated with feature[0]
  float *d_feature;
  dpct::dpct_malloc((void **)&d_feature, npoints * nfeatures * sizeof(float));
  //cudaMemcpyAsync(d_feature, features[0], npoints * nfeatures * sizeof(float), cudaMemcpyHostToDevice, 0);
  dpct::dpct_memcpy(d_feature, features[0], npoints * nfeatures * sizeof(float),
                    dpct::host_to_device);

  // d_feature_swap is written by the first kernel, and read by the second kernel 
  float *d_feature_swap;
  dpct::dpct_malloc((void **)&d_feature_swap,
                    npoints * nfeatures * sizeof(float));

  int *d_membership;
  dpct::dpct_malloc((void **)&d_membership, npoints * sizeof(int));

  sycl::range<3> grids((npoints + BLOCK_SIZE) / BLOCK_SIZE, 1, 1);
  sycl::range<3> threads(BLOCK_SIZE, 1, 1);

  /* sweep k from min to max_nclusters to find the best number of clusters */
  for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)
  {
    if (nclusters > npoints) break;	/* cannot have more clusters than points */

    int c = 0;  // for each cluster size, count the actual number of loop interations

    {
      dpct::buffer_t d_feature_swap_buf_ct0 = dpct::get_buffer(d_feature_swap);
      dpct::buffer_t d_feature_buf_ct1 = dpct::get_buffer(d_feature);
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_feature_swap_acc_ct0 =
            d_feature_swap_buf_ct0.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_feature_acc_ct1 =
            d_feature_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

        auto dpct_global_range = grids * threads;

        cgh.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(dpct_global_range.get(2),
                               dpct_global_range.get(1),
                               dpct_global_range.get(0)),
                sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              feature_transpose((float *)(&d_feature_swap_acc_ct0[0]),
                                (const float *)(&d_feature_acc_ct1[0]),
                                nfeatures, npoints, item_ct1);
            });
      });
    }

    // create clusters of size 'nclusters' (note in the previous iteration "nclusters-1", 
    // Note we free the previous cluster centers, which store the previous clusters, 
    // before using the updated one. We free the final cluster centers in the file "kmeans.cpp" 
    float** clusters;
    clusters    = (float**) malloc(nclusters *             sizeof(float*));
    clusters[0] = (float*)  malloc(nclusters * nfeatures * sizeof(float));
    for (int i=1; i<nclusters; i++) clusters[i] = clusters[i-1] + nfeatures;

    /* initialize the random clusters */
    int* initial = (int *) malloc (npoints * sizeof(int));
    for (int i = 0; i < npoints; i++) initial[i] = i;
    int initial_points = npoints;

    /* allocate the buffer before entering the nloops */
    float* d_clusters;
    dpct::dpct_malloc((void **)&d_clusters,
                      nclusters * nfeatures * sizeof(float));

    /* iterate nloops times for each number of clusters */
    for(int lp = 0; lp < nloops; lp++)
    {
      int n = 0;

      /* pick cluster centers based on the initial array 
         Maybe n = (int)rand() % initial_points; is more straightforward
         without using the initial array
       */	
      for (int i=0; i<nclusters && initial_points >= 0; i++) {

        for (int j=0; j<nfeatures; j++)
          clusters[i][j] = features[initial[n]][j];	// remapped

        /* swap the selected index with the end index. For the next iteration
           of nloops, initial[0] is differetn from initial[0] in the previous iteration */

        int temp = initial[n];
        initial[n] = initial[initial_points-1];
        initial[initial_points-1] = temp;
        initial_points--;
        n++;
      }

      /* initialize the membership to -1 for all */
      for (int i=0; i < npoints; i++) membership[i] = -1;

      /* allocate space for and initialize new_centers_len and new_centers */
      int* new_centers_len = (int*) calloc(nclusters, sizeof(int));
      float** new_centers    = (float**) malloc(nclusters *            sizeof(float*));
      new_centers[0] = (float*)  calloc(nclusters * nfeatures, sizeof(float));
      for (int i=1; i<nclusters; i++) new_centers[i] = new_centers[i-1] + nfeatures;


      /* iterate until convergence */

      int loop = 0;
      do {
        delta = 0.0;

        dpct::dpct_memcpy(d_clusters, clusters[0],
                          nclusters * nfeatures * sizeof(float),
                          dpct::host_to_device);
        {
          dpct::buffer_t d_feature_swap_buf_ct0 =
              dpct::get_buffer(d_feature_swap);
          dpct::buffer_t d_clusters_buf_ct1 = dpct::get_buffer(d_clusters);
          dpct::buffer_t d_membership_buf_ct2 = dpct::get_buffer(d_membership);
          q_ct1.submit([&](sycl::handler &cgh) {
            auto d_feature_swap_acc_ct0 =
                d_feature_swap_buf_ct0
                    .get_access<sycl::access::mode::read_write>(cgh);
            auto d_clusters_acc_ct1 =
                d_clusters_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_membership_acc_ct2 =
                d_membership_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);

            auto dpct_global_range = grids * threads;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(threads.get(2), threads.get(1),
                                                 threads.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                  find_membership((const float *)(&d_feature_swap_acc_ct0[0]),
                                  (const float *)(&d_clusters_acc_ct1[0]),
                                  (int *)(&d_membership_acc_ct2[0]), nclusters,
                                  nfeatures, npoints, item_ct1);
                });
          });
        }
        dpct::dpct_memcpy(membership_OCL, d_membership, npoints * sizeof(int),
                          dpct::device_to_host);

        /* 
           1 compute the 'new' size and center of each cluster 
           2 update the membership of each point. 
         */

        for (int i = 0; i < npoints; i++)
        {
          int cluster_id = membership_OCL[i];
          new_centers_len[cluster_id]++;
          if (membership_OCL[i] != membership[i])
          {
            delta++;
            membership[i] = membership_OCL[i];
          }
          for (int j = 0; j < nfeatures; j++)
          {
            new_centers[cluster_id][j] += features[i][j];
          }
        }

        /* replace old cluster centers with new_centers */
        for (int i=0; i<nclusters; i++) {
          //printf("length of new cluster %d = %d\n", i, new_centers_len[i]);
          for (int j=0; j<nfeatures; j++) {
            if (new_centers_len[i] > 0)
              clusters[i][j] = new_centers[i][j] / new_centers_len[i];	/* take average i.e. sum/n */
            new_centers[i][j] = 0.0; // reset 
          }
          new_centers_len[i] = 0; // reset
        }
        c++;
      } while ((delta > threshold) && (loop++ < 500));	/* makes sure loop terminates */

      free(new_centers[0]);
      free(new_centers);
      free(new_centers_len);

      /* find the number of clusters with the best RMSE (enabled with the cmd-line option -r) */
      if(isRMSE)
      {
        rmse = rms_err(features,
            nfeatures,
            npoints,
            clusters,
            nclusters);
        //printf("#cluster=%d rmse=%f\n", nclusters, rmse);

        if(rmse < min_rmse_ref){
          min_rmse_ref = rmse;			//update reference min RMSE
          *min_rmse = min_rmse_ref;		//update return min RMSE
          *best_nclusters = nclusters;	//update optimum number of clusters
          index = lp;						//update number of iteration to reach best RMSE
        }
      }
    } // for(int lp = 0; lp < nloops; lp++)

    // free the previous cluster centers before using the updated one for each cluster size
    if (*cluster_centres) {
      free((*cluster_centres)[0]);
      free(*cluster_centres);
    }
    *cluster_centres = clusters;

    dpct::dpct_free(d_clusters);
    free(initial);
  } // for(int nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++)

  dpct::dpct_free(d_feature);
  dpct::dpct_free(d_feature_swap);
  dpct::dpct_free(d_membership);

  free(membership_OCL);
  free(membership);

  return index;
}

