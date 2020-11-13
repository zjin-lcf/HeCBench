#ifndef _KERNELS_COMMON_H_
#define _KERNELS_COMMON_H_

#pragma once
#pragma warning(disable:4996)

#define _USE_MATH_DEFINES
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "tuningParameters.h"
#include "qtc_common.h"

// Forward declarations
void QTC_device( float *dist_matrix, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, int *ungrpd_pnts_indr, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int cwrank, int node_rank, int node_count, int total_thread_block_count);

int generate_candidate_cluster_compact_storage(int seed_point, int degree, char *Ai_mask, float *compact_storage_dist_matrix, char *clustered_pnts_mask, int *indr_mtrx, float *dist_to_clust, int point_count, int N0, int max_degree, int *candidate_cluster, float threshold,
                                               sycl::nd_item<3> item_ct1,
                                               float *dist_array,
                                               int *point_index_array);

int find_closest_point_to_cluster(int seed_point, int latest_point, char *Ai_mask, char *clustered_pnts_mask, float *work, int *indr_mtrx, float *dist_to_clust, int pointCount, int N0, int max_degree, float threshold);

void QTC(const string& name, OptionParser& op, int matrix_type);


//
// arrange blocks into 2D grid that fits into the GPU ( for powers of two only )
//
inline sycl::range<3> grid2D(int nblocks)
{
    int slices = 1;

    if( nblocks < 1 )
      return sycl::range<3>(1, 1, 1);

    while( nblocks/slices > 65535 )
        slices *= 2;
   return sycl::range<3>(nblocks / slices, slices, 1);
}


inline 
int closest_point_reduction(float min_dist, float threshold, int closest_point,
                            sycl::nd_item<3> item_ct1, float *dist_array,
                            int *point_index_array){

   int tid = item_ct1.get_local_id(2);
   int curThreadCount = item_ct1.get_local_range().get(2) *
                        item_ct1.get_local_range().get(1) *
                        item_ct1.get_local_range().get(0);

    dist_array[tid] = min_dist;
    point_index_array[tid] = closest_point;

   item_ct1.barrier();

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

   item_ct1.barrier();

    return point_index_array[0];
}



void reduce_card_device(int *cardinalities, int TB_count){
    int i, max_card = -1, winner_index;

    for(i=0; i<TB_count*2; i+=2){
        if( cardinalities[i] > max_card ){
            max_card = cardinalities[i];
            winner_index = cardinalities[i+1];
        }
    }

    cardinalities[0] = max_card;
    cardinalities[1] = winner_index;

}



void
compute_degrees(int *indr_mtrx, int *degrees, int N0, int max_degree,
                sycl::nd_item<3> item_ct1){
    int tid, tblock_id, TB_count, offset;
    int local_point_count, curThreadCount;
    int starting_point;

   curThreadCount = item_ct1.get_local_range().get(2);
   tid = item_ct1.get_local_id(2);
   tblock_id = item_ct1.get_group(2);
   TB_count = item_ct1.get_group_range(2);
    local_point_count = (N0+TB_count-1)/TB_count;
    starting_point = tblock_id * local_point_count;
    offset =  starting_point*max_degree;
    indr_mtrx = &indr_mtrx[offset];
    degrees = &degrees[starting_point];

    // The last threadblock might end up with less points.
    if( (tblock_id+1)*local_point_count > N0 )
        local_point_count = MAX(0,N0-starting_point);

    for(int i=0; i+tid < local_point_count; i+=curThreadCount){
        int cnt = 0;
        for(int j=0; j < max_degree; j++){
            if( indr_mtrx[(i+tid)*max_degree+j] >= 0 ){
                ++cnt;
            }
        }
        degrees[i+tid] = cnt;
    }
}

/*
__global__ void
compute_degrees(int *indr_mtrx, int *degrees, int N0, int max_degree){
    int tid, tblock_id, TB_count, offset;
    int local_point_count, curThreadCount;
    int starting_point;

    curThreadCount = blockDim.x*blockDim.y*blockDim.z;
    tid = threadIdx.x;
    tblock_id = (blockIdx.y * gridDim.x + blockIdx.x);
    TB_count = gridDim.y * gridDim.x;
    local_point_count = (N0+TB_count-1)/TB_count;
    starting_point = tblock_id * local_point_count;
    offset =  starting_point*max_degree;
    indr_mtrx = &indr_mtrx[offset];
    degrees = &degrees[starting_point];

    // The last threadblock might end up with less points.
    if( (tblock_id+1)*local_point_count > N0 )
        local_point_count = MAX(0,N0-starting_point);

    for(int i=0; i+tid < local_point_count; i+=curThreadCount){
        int cnt = 0;
        for(int j=0; j < max_degree; j++){
            if( indr_mtrx[(i+tid)*max_degree+j] >= 0 ){
                ++cnt;
            }
        }
        degrees[i+tid] = cnt;
    }
}
*/

void
update_clustered_pnts_mask(char *clustered_pnts_mask, char *Ai_mask, int N0 ,
                           sycl::nd_item<3> item_ct1) {
   int tid = item_ct1.get_local_id(2);
   int curThreadCount = item_ct1.get_local_range().get(2) *
                        item_ct1.get_local_range().get(1) *
                        item_ct1.get_local_range().get(0);

    // If a point is part of the latest winner cluster, then it should be marked as
    // clustered for the future iterations. Otherwise it should be left as it is.
    for(int i = 0; i+tid < N0; i+=curThreadCount){
        clustered_pnts_mask[i+tid] |= Ai_mask[i+tid];
    }
   item_ct1.barrier();
}


void
trim_ungrouped_pnts_indr_array(int seed_index, int *ungrpd_pnts_indr, float *dist_matrix, int *result_cluster, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold,
                               sycl::nd_item<3> item_ct1, float *dist_array,
                               int *point_index_array, int *tmp_pnts,
                               int *cnt_sh, bool *flag_sh) {
    int cnt;
   int tid = item_ct1.get_local_id(2);
   int curThreadCount = item_ct1.get_local_range().get(2) *
                        item_ct1.get_local_range().get(1) *
                        item_ct1.get_local_range().get(0);

    int degree = degrees[seed_index];
   (void)generate_candidate_cluster_compact_storage(
       seed_index, degree, Ai_mask, dist_matrix, clustered_pnts_mask, indr_mtrx,
       dist_to_clust, point_count, N0, max_degree, result_cluster, threshold,
       item_ct1, dist_array, point_index_array);

    if( 0 == tid ){
      *cnt_sh = 0;
      *flag_sh = false;
    }
   item_ct1.barrier();

    for(int i = 0; i+tid < point_count; i+=curThreadCount){
        // Have all threads make a coalesced read of contiguous global memory and copy the points assuming they are all good.
        tmp_pnts[tid] = ungrpd_pnts_indr[i+tid];
        int pnt = tmp_pnts[tid];
        // If a point is bad (which should not happen very often), raise a global flag so that thread zero fixes the problem.
        if( 1 == Ai_mask[pnt] ){
         *flag_sh = true;
            tmp_pnts[tid] = INVALID_POINT_MARKER;
        }else{
         ungrpd_pnts_indr[*cnt_sh + tid] = pnt;
        }

      item_ct1.barrier();

        if( 0 == tid ){
         if ((*flag_sh)) {
            cnt = *cnt_sh;
                for(int j = 0; (j < curThreadCount) && (i+j < point_count); j++ ){
                    if( INVALID_POINT_MARKER != tmp_pnts[j] ){
                        ungrpd_pnts_indr[cnt] = tmp_pnts[j];
                        cnt++;
                    }
                }
            *cnt_sh = cnt;
            }else{
            *cnt_sh += curThreadCount;
            }
         *flag_sh = false;
        }

      item_ct1.barrier();
    }
}



void QTC_device( float *dist_matrix, char *Ai_mask, char *clustered_pnts_mask, int *indr_mtrx, int *cluster_cardinalities, int *ungrpd_pnts_indr, float *dist_to_clust, int *degrees, int point_count, int N0, int max_degree, float threshold, int node_rank, int node_count, int total_thread_block_count,
                 sycl::nd_item<3> item_ct1, float *dist_array,
                 int *point_index_array) {
    int max_cardinality = -1;
    int max_cardinality_index;
    int i, tblock_id, tid, base_offset;

   tid = item_ct1.get_local_id(2);
    //tblock_id = (blockIdx.y * gridDim.x + blockIdx.x);
   tblock_id = item_ct1.get_group(2);
    Ai_mask = &Ai_mask[tblock_id * N0];
    dist_to_clust = &dist_to_clust[tblock_id * max_degree];
    //base_offset = node_offset+tblock_id;
    base_offset = tblock_id*node_count + node_rank;

    // for i loop of the algorithm.
    // Each thread iterates over all points that the whole thread-block owns
    for(i = base_offset; i < point_count; i+= total_thread_block_count ){
        int cnt;
        int seed_index = ungrpd_pnts_indr[i];
        int degree = degrees[seed_index];
        if( degree <= max_cardinality ) continue;
      cnt = generate_candidate_cluster_compact_storage(
          seed_index, degree, Ai_mask, dist_matrix, clustered_pnts_mask,
          indr_mtrx, dist_to_clust, point_count, N0, max_degree, NULL,
          threshold, item_ct1, dist_array, point_index_array);
        if( cnt > max_cardinality ){
            max_cardinality = cnt;
            max_cardinality_index = seed_index;
        }
    } // for (i

    // since only three elements per block go to the global memory, the offset is:
    //int card_offset = (blockIdx.y * gridDim.x + blockIdx.x)*2;
   int card_offset = item_ct1.get_group(2) * 2;
    // only one thread needs to write into the global memory since they all have the same information.
    if( 0 == tid ){
        cluster_cardinalities[card_offset] = max_cardinality;
        cluster_cardinalities[card_offset+1] = max_cardinality_index;
    }
}

#endif
