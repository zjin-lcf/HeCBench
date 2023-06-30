#ifndef _KERNELS_COMPACT_STORAGE_H_
#define _KERNELS_COMPACT_STORAGE_H_

#define COMPUTE_DIAMETER_WITH_POINT( _CAND_PNT_, _CURR_DIST_TO_CLUST_, _I_ ) \
    if( (_CAND_PNT_) < 0 ){\
        break;\
    }\
do{\
    int tmp_index = (_I_)*curThreadCount+tid;\
    if( (_CAND_PNT_) == seed_point ){\
        break;\
    }\
    _CURR_DIST_TO_CLUST_ = dist_to_clust[ tmp_index ];\
    /* if "_CAND_PNT_" is too far away, or already in Ai_mask, or in clustered_points, ignore it. */\
    if( (_CURR_DIST_TO_CLUST_ > threshold) || (0 != Ai_mask[(_CAND_PNT_)]) || (0 != clustered_pnts_mask[(_CAND_PNT_)]) ){ \
        _CAND_PNT_ = seed_point; /* This is so we don't do the lookup again. */\
        break;\
    }\
    dist_to_new_point = threshold+1;\
    /* Find _CAND_PNT_ in the neighborhood of the latest_point.*/\
    for(int j=last_index_checked; j<max_degree; j++){\
        int tmp_pnt = indr_mtrx[ latest_p_off + j ];\
        if( (tmp_pnt > (_CAND_PNT_)) || (tmp_pnt < 0) ){\
            last_index_checked = j;\
            break;\
        }\
        if( tmp_pnt == (_CAND_PNT_) ){\
                dist_to_new_point = compact_storage_dist_matrix[ latest_p_off + j ];\
            break;\
        }\
    }\
\
    /* See if the distance of "_CAND_PNT_" to the "latest_point" is larger */\
    /* than the previous, cached distance of "_CAND_PNT_" to the cluster.  */\
    if(dist_to_new_point > _CURR_DIST_TO_CLUST_){\
        diameter = dist_to_new_point;\
        dist_to_clust[ tmp_index ] = diameter;\
    }else{\
        diameter = _CURR_DIST_TO_CLUST_;\
    }\
\
    /* The point that leads to the cluster with the smallest diameter is the closest point */\
    if( diameter < min_dist ){\
        min_dist = diameter;\
        point_index = (_CAND_PNT_);\
    }\
}while(0)



#define FETCH_POINT( _CAND_PNT_ , _I_ )\
{\
    int tmp_index = (_I_)*curThreadCount+tid;\
    if( tmp_index >= max_degree ){\
        break;\
    }\
    _CAND_PNT_ = indr_mtrx[ seed_p_off + tmp_index ];\
    if( (_CAND_PNT_) < 0 ){\
        break;\
    }\
}

int generate_candidate_cluster_compact_storage(
    sycl::nd_item<1> &item, 
    float *dist_array,
    int *point_index_array,
    const int seed_point, 
    const int degree, 
    char *Ai_mask, 
    float *compact_storage_dist_matrix, 
    char *clustered_pnts_mask, 
    int *indr_mtrx, 
    float *dist_to_clust, 
    const int point_count,
    const int N0,
    const int max_degree, 
    int *candidate_cluster, 
    const float threshold)
{

  bool flag;
  int cnt, latest_point;
  int tid = item.get_local_id(0);
  int curThreadCount  = item.get_local_range(0);

  int seed_p_off;

  float curr_dist_to_clust_i;
  float curr_dist_to_clust_0, curr_dist_to_clust_1, curr_dist_to_clust_2, curr_dist_to_clust_3;
  float curr_dist_to_clust_4, curr_dist_to_clust_5, curr_dist_to_clust_6, curr_dist_to_clust_7;
  float curr_dist_to_clust_8, curr_dist_to_clust_9, curr_dist_to_clust_10, curr_dist_to_clust_11;
  int cand_pnt_i=-1;
  int cand_pnt_0=-1, cand_pnt_1=-1, cand_pnt_2=-1, cand_pnt_3=-1;
  int cand_pnt_4=-1, cand_pnt_5=-1, cand_pnt_6=-1, cand_pnt_7=-1;
  int cand_pnt_8=-1, cand_pnt_9=-1, cand_pnt_10=-1, cand_pnt_11=-1;

  // Cleanup the candidate-cluster-mask, Ai_mask
  for(int i=0; i+tid < N0; i+=curThreadCount){
    Ai_mask[i+tid] = 0;
  }

  // Cleanup the "distance cache"
  for(int i=0; i+tid < max_degree; i+=curThreadCount){
    dist_to_clust[i+tid] = 0;
  }

  // Put the seed point in the candidate cluster and mark it as taken in the candidate cluster mask Ai_mask.
  flag = true;
  cnt = 1;
  if( 0 == tid ){
    if( NULL != candidate_cluster )
      candidate_cluster[0] = seed_point;
    Ai_mask[seed_point] = 1;
  }
  item.barrier(sycl::access::fence_space::local_space);
  seed_p_off = seed_point*max_degree;
  latest_point = seed_point;

  // Prefetch 12 points per thread, into registers, to reduce the memory pressure (and delay) of
  // constantly going to memory to fetch these points inside the while() loop that follows.
  do{
    FETCH_POINT(  cand_pnt_0,  0 );
    FETCH_POINT(  cand_pnt_1,  1 );
    FETCH_POINT(  cand_pnt_2,  2 );
    FETCH_POINT(  cand_pnt_3,  3 );
    FETCH_POINT(  cand_pnt_4,  4 );
    FETCH_POINT(  cand_pnt_5,  5 );
    FETCH_POINT(  cand_pnt_6,  6 );
    FETCH_POINT(  cand_pnt_7,  7 );
    FETCH_POINT(  cand_pnt_8,  8 );
    FETCH_POINT(  cand_pnt_9,  9 );
    FETCH_POINT( cand_pnt_10, 10 );
    FETCH_POINT( cand_pnt_11, 11 );
  }while(0);

  // different threads might exit this loop at different times, so let them catch up.
  item.barrier(sycl::access::fence_space::local_space);

  while( (cnt < point_count) && flag ){
    int point_index = -1;
    float min_dist=3*threshold;
    int last_index_checked = 0;
    float diameter;
    float dist_to_new_point;
    int latest_p_off = latest_point*max_degree;

    do{
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_0,  curr_dist_to_clust_0,  0 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_1,  curr_dist_to_clust_1,  1 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_2,  curr_dist_to_clust_2,  2 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_3,  curr_dist_to_clust_3,  3 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_4,  curr_dist_to_clust_4,  4 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_5,  curr_dist_to_clust_5,  5 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_6,  curr_dist_to_clust_6,  6 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_7,  curr_dist_to_clust_7,  7 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_8,  curr_dist_to_clust_8,  8 );
      COMPUTE_DIAMETER_WITH_POINT(  cand_pnt_9,  curr_dist_to_clust_9,  9 );
      COMPUTE_DIAMETER_WITH_POINT( cand_pnt_10, curr_dist_to_clust_10, 10 );
      COMPUTE_DIAMETER_WITH_POINT( cand_pnt_11, curr_dist_to_clust_11, 11 );
    }while(0);

    // different threads might exit this loop at different times, so let them catch up.
    item.barrier(sycl::access::fence_space::local_space);

    // The following loop implements the "find point pj s.t. diameter(Ai && pj) is minimum"
    for(int i=12; i*curThreadCount+tid < max_degree; i++){
      FETCH_POINT( cand_pnt_i, i );
      COMPUTE_DIAMETER_WITH_POINT( cand_pnt_i, curr_dist_to_clust_i, i );
    }
    item.barrier(sycl::access::fence_space::local_space);

    //min_G_index = closest_point_reduction(min_dist, threshold, point_index);
    dist_array[tid] = min_dist;
    point_index_array[tid] = point_index;
    item.barrier(sycl::access::fence_space::local_space);

    if(tid == 0 ){
      for(int j=1; j<curThreadCount; j++){
        float dist = dist_array[j];
        // look for a point that is closer, or equally far, but with a smaller index.
        if( (dist < min_dist) || (dist == min_dist && point_index_array[j] < point_index_array[0]) ){
          min_dist = dist;
          point_index_array[0] = point_index_array[j];
        }
      }
      if( min_dist > threshold )
        point_index_array[0] = -1;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int min_G_index = point_index_array[0];

    if(min_G_index >= 0 ){
      if( 0 == tid ){
        Ai_mask[min_G_index] = 1;
        if( NULL != candidate_cluster ){
          candidate_cluster[cnt] = min_G_index;
        }
      }
      latest_point = min_G_index;
      cnt++;
    }else{
      flag = false;
    }

    item.barrier(sycl::access::fence_space::local_space);
  }
  item.barrier(sycl::access::fence_space::local_space);

  return cnt;
}


#endif
