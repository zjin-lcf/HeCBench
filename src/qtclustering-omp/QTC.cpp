#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <chrono>
#include <sstream>
#include <iostream>
#include <fstream>
#include <omp.h>
#include "tuningParameters.h"
#include "QTC.h"
#include "OptionParser.h"
#include "libdata.h"

#define _USE_MATH_DEFINES
#include <float.h>
#include "comm.h"

using namespace std;

#include "kernels_compact_storage.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in megabytes if they are not using a
//   predefined size (i.e. the -s option).
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op){
  op.addOption("PointCount", OPT_INT, "4096", "point count (default: 4096)");
  op.addOption("Threshold", OPT_FLOAT, "1", "cluster diameter threshold (default: 1)");
  op.addOption("SaveOutput", OPT_BOOL, "", "Save output results in files (default: false)");
  op.addOption("Verbose", OPT_BOOL, "", "Print cluster cardinalities (default: false)");
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Calls single precision and, if viable, double precision QT-Clustering
//   benchmark.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************
void runTest(const string& name, OptionParser& op);

void RunBenchmark(OptionParser &op){
  runTest("QTC", op);
}

// ****************************************************************************
// Function: calculate_participants
//
// Purpose:
//   This function decides how many GPUs (up to the maximum requested by the user)
//   and threadblocks per GPU will be used. It also returns the total number of
//   thread-blocks across all GPUs and the number of thread-blocks that are in nodes
//   before the current one.
//   In the future, the behavior of this function should be decided based on
//   auto-tuning instead of arbitrary decisions.
//
// Arguments:
//   The number of nodes requested by the user and the four
//   variables that the function computes (passed by reference)
//
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: May 25, 2011
//
// ****************************************************************************
void calculate_participants(int point_count, int node_count, int cwrank, 
    int *thread_block_count, int *total_thread_block_count, int *active_node_count){

  int ac_nd_cnt, thr_blc_cnt, total_thr_blc_cnt;

  ac_nd_cnt = node_count;
  if( point_count <= (node_count-1) * SM_COUNT * GPU_MIN_SATURATION_FACTOR ){
    int K = SM_COUNT * GPU_MIN_SATURATION_FACTOR;
    ac_nd_cnt = (point_count+K-1) / K;
  }

  if( point_count >= ac_nd_cnt * SM_COUNT * OVR_SBSCR_FACTOR ){
    thr_blc_cnt = SM_COUNT * OVR_SBSCR_FACTOR;
    total_thr_blc_cnt = thr_blc_cnt * ac_nd_cnt;
  }else{
    thr_blc_cnt = point_count/ac_nd_cnt;
    if( cwrank < point_count%ac_nd_cnt ){
      thr_blc_cnt++;
    }
    total_thr_blc_cnt = point_count;
  }

  *active_node_count  = ac_nd_cnt;

  return;
}

// ****************************************************************************
// Function: runTest
//
// Purpose:
//   This benchmark measures the performance of applying QT-clustering on
//   single precision data.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: February 04, 2011
//
// ****************************************************************************

void runTest(const string& name, OptionParser& op)
{
  int matrix_type = 0x0;
  if( 0 == comm_get_rank() ){
    matrix_type |= GLOBAL_MEMORY;
    matrix_type |= COMPACT_STORAGE_MATRIX;
  }
  comm_broadcast ( &matrix_type, 1, COMM_TYPE_INT, 0);

  QTC(name, op, matrix_type);

}

////////////////////////////////////////////////////////////////////////////////
//
void QTC(const string& name, OptionParser& op, int matrix_type){
  ofstream debug_out, seeds_out;
  int *indr_mtrx_host, *ungrpd_pnts_indr_host;
  bool save_clusters = false;
  bool be_verbose = false;
  float *dist_source, *pnts;
  float threshold = 1.0f;
  int i, max_degree, thread_block_count, total_thread_block_count, active_node_count;
  int cwrank=0, node_count=1, tpb, max_card, iter=0;
  unsigned long int dst_matrix_elems, point_count, max_point_count;

  point_count = op.getOptionInt("PointCount");
  threshold = op.getOptionFloat("Threshold");
  save_clusters = op.getOptionBool("SaveOutput");
  be_verbose = op.getOptionBool("Verbose");


  // TODO - only deal with this size-switch once
  int def_size = op.getOptionInt("size");
  switch( def_size ) {
    case 1:
      // size == 1 should match default values of PointCount,
      // Threshold, TextureMem, and CompactStorage parameters.
      // (i.e., -s 1 is the default)
      point_count    = 4*1024;
      break;
    case 2:
      point_count    = 8*1024;
      break;
    case 3:
      point_count    = 16*1024;
      break;
    case 4:
      point_count    = 16*1024;
      break;
    case 5:
      point_count    = 26*1024;
      break;
    default:
      fprintf( stderr, "unsupported size %d given; terminating\n", def_size );
      return;
  }

  cwrank = comm_get_rank();
  node_count = comm_get_size();

  if( cwrank == 0 ){
    pnts = generate_synthetic_data(&dist_source, &indr_mtrx_host, &max_degree, threshold, point_count, matrix_type);
  }

  comm_broadcast ( &point_count, 1, COMM_TYPE_INT, 0);
  comm_broadcast ( &max_degree, 1, COMM_TYPE_INT, 0);

  dst_matrix_elems = point_count*max_degree;

  if( cwrank != 0 ){ // For all nodes except zero, in a distributed run.
    dist_source = (float*) malloc (sizeof(float)*dst_matrix_elems);
    indr_mtrx_host = (int*) malloc (sizeof(int)*point_count*max_degree);
  }
  // If we need to print the actual clusters later on, we'll need to have all points in all nodes.
  if( save_clusters ){
    if( cwrank != 0 ){
      pnts = (float *)malloc( 2*point_count*sizeof(float) );
    }
    comm_broadcast ( pnts, 2*point_count, COMM_TYPE_FLOAT, 0);
  }

  comm_broadcast ( dist_source, dst_matrix_elems, COMM_TYPE_FLOAT, 0);
  comm_broadcast ( indr_mtrx_host, point_count*max_degree, COMM_TYPE_INT, 0);

  assert( max_degree > 0 );

  calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

  ungrpd_pnts_indr_host = (int*) malloc (sizeof(int)*point_count);
  for(int i=0; i<point_count; i++){
    ungrpd_pnts_indr_host[i] = i;
  }

  int* cardnl = (int*) malloc (sizeof(int)*thread_block_count*2);
  int* result = (int*) malloc (sizeof(int)*point_count);
  int* degrees = (int*) malloc (sizeof(int)*point_count);
  char* Ai_mask = (char*) malloc (sizeof(char)*thread_block_count*point_count);
  float* dist_to_clust = (float*) malloc (sizeof(float)*thread_block_count*max_degree);
  char* clustered_pnts_mask = (char*) malloc (sizeof(char)*point_count);

  //allocDeviceBuffer(&degrees,             point_count*sizeof(int));
  //allocDeviceBuffer(&ungrpd_pnts_indr,    point_count*sizeof(int));
  //allocDeviceBuffer(&Ai_mask,             thread_block_count*point_count*sizeof(char));
  //allocDeviceBuffer(&dist_to_clust,       thread_block_count*max_degree*sizeof(float));
  //allocDeviceBuffer(&clustered_pnts_mask, point_count*sizeof(char));

  // This is the N*Delta indirection matrix
  //allocDeviceBuffer(&distance_matrix_gmem, dst_matrix_elems*sizeof(float));
  //allocDeviceBuffer(&indr_mtrx, point_count*max_degree*sizeof(int));
  //allocDeviceBuffer(&degrees,             point_count*sizeof(int));
  //allocDeviceBuffer(&ungrpd_pnts_indr,    point_count*sizeof(int));
  //allocDeviceBuffer(&Ai_mask,             thread_block_count*point_count*sizeof(char));
  //allocDeviceBuffer(&dist_to_clust,       thread_block_count*max_degree*sizeof(float));
  //allocDeviceBuffer(&clustered_pnts_mask, point_count*sizeof(char));
  //allocDeviceBuffer(&cardnl,              thread_block_count*2*sizeof(int));
  //allocDeviceBuffer(&result,              point_count*sizeof(int));

#pragma omp target data map(to: dist_source[0:dst_matrix_elems], \
                                indr_mtrx_host[0:point_count*max_degree], \
                                ungrpd_pnts_indr_host[0:point_count]) \
  map(alloc: degrees[0:point_count], \
              Ai_mask[0:thread_block_count*point_count], \
              dist_to_clust[0:max_degree*thread_block_count], \
              clustered_pnts_mask[0:point_count], \
              cardnl[0:thread_block_count*2], \
              result[0:point_count])
  {
    //cudaMemset(clustered_pnts_mask, 0, point_count*sizeof(char));
    #pragma omp target teams distribute parallel for 
    for (int i = 0; i < point_count; i++)
      clustered_pnts_mask[i] = 0;

    //cudaMemset(dist_to_clust, 0, max_degree*thread_block_count*sizeof(float));
    #pragma omp target teams distribute parallel for 
    for (int i = 0; i < max_degree*thread_block_count; i++)
      dist_to_clust[i] = 0;

    tpb = ( point_count > THREADSPERBLOCK )? THREADSPERBLOCK : point_count;

    printf("compute degrees\n");
    //compute_degrees<<<grid2D(thread_block_count), tpb>>>((int *)indr_mtrx, (int *)degrees, point_count, max_degree);

    #pragma omp target teams num_teams(thread_block_count) thread_limit(tpb)
    {
      #pragma omp parallel 
      {
        int curThreadCount = omp_get_num_threads();
        int tid = omp_get_thread_num();
        int tblock_id = omp_get_team_num();
        int TB_count = omp_get_num_teams();
        int local_point_count = (point_count+TB_count-1)/TB_count;
        int starting_point = tblock_id * local_point_count;
        int offset =  starting_point*max_degree;
        int *indr = &indr_mtrx_host[offset];
        int *degree = &degrees[starting_point];

        // The last threadblock might end up with less points.
        for(int i=0; i+tid < local_point_count; i+=curThreadCount){
          int cnt = 0;
          for(int j=0; j < max_degree; j++){
            if( indr[(i+tid)*max_degree+j] >= 0 ){
              ++cnt;
            }
          }
          degree[i+tid] = cnt;
        }
      }
    }

    const char *sizeStr;
    stringstream ss;
    ss << "PointCount=" << (long)point_count;
    sizeStr = strdup(ss.str().c_str());

    // The names of the saved outputs, if enabled, are "p", "p_seeds", and "p."
    if( 0 == cwrank ){
      if( save_clusters ){
        debug_out.open("p");
        for(i=0; i<point_count; i++){
          debug_out << pnts[2*i] << " " << pnts[2*i+1] << std::endl;
        }
        debug_out.close();
        seeds_out.open("p_seeds");
      }

      cout << "\nInitial ThreadBlockCount: " << thread_block_count;
      cout << " PointCount: " << point_count;
      cout << " Max degree: " << max_degree << "\n" << std::endl;
      cout.flush();
    }

    max_point_count = point_count;

    tpb = THREADSPERBLOCK;

    // Kernel execution
    double qtc_time = 0.0, trim_time = 0.0, update_time = 0.0;

    do{
      stringstream ss;
      int winner_node=-1;
      int winner_index=-1;
      bool this_node_participates = true;

      ++iter;

      calculate_participants(point_count, node_count, cwrank, &thread_block_count, &total_thread_block_count, &active_node_count);

      // If there are only a few elements left to cluster, reduce the number of participating nodes (GPUs).
      if( cwrank >= active_node_count ){
        this_node_participates = false;
      }
      comm_update_communicator(cwrank, active_node_count);
      if( !this_node_participates )
        break;
      cwrank = comm_get_rank();

      auto start = std::chrono::steady_clock::now();

      //QTC_device<<<grid, tpb>>>((float*)distance_matrix, (char *)Ai_mask, (char *)clustered_pnts_mask,
      //(int *)indr_mtrx, (int *)cardnl, (int *)ungrpd_pnts_indr,
      //(float *)dist_to_clust, (int *)degrees, point_count, max_point_count,
      //max_degree, threshold, cwrank, active_node_count,
      //total_thread_block_count);

      #pragma omp target teams num_teams(thread_block_count) thread_limit(tpb)
      {
        float dist_array[THREADSPERBLOCK];
        int point_index_array[THREADSPERBLOCK];
        #pragma omp parallel 
        {
          int max_cardinality = -1;
          int max_cardinality_index;

          int tid = omp_get_thread_num();
          int tblock_id = omp_get_team_num();
          char *Ai_mask_ptr = &Ai_mask[tblock_id * max_point_count];
          float *dist_to_clust_ptr = &dist_to_clust[tblock_id * max_degree];
          int base_offset = tblock_id*node_count + cwrank;

          // for i loop of the algorithm.
          // Each thread iterates over all points that the whole thread-block owns
          for(int i = base_offset; i < point_count; i+= total_thread_block_count ){
            int seed_index = ungrpd_pnts_indr_host[i];
            int degree = degrees[seed_index];
            if( degree <= max_cardinality ) continue;
            int  cnt = generate_candidate_cluster_compact_storage( 
                dist_array, point_index_array,
                seed_index, degree, Ai_mask_ptr, 
                dist_source,
                clustered_pnts_mask,
                indr_mtrx_host,
                dist_to_clust_ptr,
                point_count, max_point_count, max_degree, NULL, threshold);
            if( cnt > max_cardinality ){
              max_cardinality = cnt;
              max_cardinality_index = seed_index;
            }
          } // for (i

          // since only three elements per block go to the global memory, the offset is:
          int card_offset = tblock_id*2;
          // only one thread needs to write into the global memory since they all have the same information.
          if( 0 == tid ){
            cardnl[card_offset] = max_cardinality;
            cardnl[card_offset+1] = max_cardinality_index;
          }
        }
      }

      auto end = std::chrono::steady_clock::now();
      qtc_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

#ifdef DEBUG
      printf("iteration %d: cardinalities\n", iter);
      #pragma omp target update from (cardnl[0:thread_block_count*2])
      for (int i = 0; i < 576*2; i++)
        printf("%d %d\n", i, cardnl[i]);
#endif

      if( thread_block_count > 1 ){
        // We are reducing 128 numbers or less, so one thread should be sufficient.
        //reduce_card_device<<<grid2D(1), 1>>>((int *)cardnl, thread_block_count);
        printf("reduce_card\n");
#pragma omp target 
        {
          int max_card = -1;
          int  winner_index;
          for(int i=0; i<thread_block_count*2; i+=2){
            if( cardnl[i] > max_card ){
              max_card = cardnl[i];
              winner_index = cardnl[i+1];
            }
          }
          cardnl[0] = max_card;
          cardnl[1] = winner_index;
        }
      }

      //copyFromDevice( cardinalities, cardnl, 2*sizeof(int) );
      #pragma omp target update from (cardnl[0:2])

      max_card     = cardnl[0];
      winner_index = cardnl[1];

      comm_barrier();

      comm_find_winner(&max_card, &winner_node, &winner_index, cwrank, max_point_count+1);

      if( be_verbose && cwrank == winner_node){ // for non-parallel cases, both "cwrank" and "winner_node" should be zero.
        cout << "[" << cwrank << "] Cluster Cardinality: " << max_card << " (Node: " << cwrank << ", index: " << winner_index << ")" << std::endl;
      }

      //trim_ungrouped_pnts_indr_array<<<grid2D(1), tpb>>>(winner_index, (int*)ungrpd_pnts_indr, (float*)distance_matrix,
      //   (int *)result, (char *)Ai_mask, (char *)clustered_pnts_mask,
      //  (int *)indr_mtrx, (int *)cardnl, (float *)dist_to_clust, (int *)degrees,
      // point_count, max_point_count, max_degree, threshold);
      //printf("point count: %d\n", point_count);

      start = std::chrono::steady_clock::now();

      #pragma omp target teams num_teams(1) thread_limit(tpb)
      {
        int tmp_pnts[THREADSPERBLOCK];
        int cnt_sh;
        bool flag_sh;
        float dist_array[THREADSPERBLOCK];
        int point_index_array[THREADSPERBLOCK];
        #pragma omp parallel 
        {
          int cnt;
          int tid = omp_get_thread_num();
          int curThreadCount = omp_get_num_threads();
          int degree = degrees[winner_index];

          generate_candidate_cluster_compact_storage( 
              dist_array, point_index_array,
              winner_index, degree, Ai_mask,
              dist_source,
              clustered_pnts_mask,
              indr_mtrx_host,
              dist_to_clust,
              point_count, max_point_count, max_degree, 
              result, threshold);


          if( 0 == tid ){
            cnt_sh = 0;
            flag_sh = false;
          }
          #pragma omp barrier

          for(int i = 0; i+tid < point_count; i+=curThreadCount){
            // Have all threads make a coalesced read of contiguous global memory and copy the points assuming they are all good.
            tmp_pnts[tid] = ungrpd_pnts_indr_host[i+tid];
            int pnt = tmp_pnts[tid];
            // If a point is bad (which should not happen very often), raise a global flag so that thread zero fixes the problem.
            if( 1 == Ai_mask[pnt] ){
              flag_sh = true;
              tmp_pnts[tid] = INVALID_POINT_MARKER;
            }else{
              ungrpd_pnts_indr_host[cnt_sh+tid] = pnt;
            }

            #pragma omp barrier

            if( 0 == tid ){
              if( flag_sh ){
                cnt = cnt_sh;
                for(int j = 0; (j < curThreadCount) && (i+j < point_count); j++ ){
                  if( INVALID_POINT_MARKER != tmp_pnts[j] ){
                    ungrpd_pnts_indr_host[cnt] = tmp_pnts[j];
                    cnt++;
                  }
                }
                cnt_sh = cnt;
              }else{
                cnt_sh += curThreadCount;
              }
              flag_sh  = false;
            }
            #pragma omp barrier
          }
        }
      }

      end = std::chrono::steady_clock::now();
      trim_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      if( cwrank == winner_node){ // for non-parallel cases, these should both be zero.
        if( save_clusters ){
          ss << "p." << iter;
          debug_out.open(ss.str().c_str());
        }

        //copyFromDevice(output, (void *)result, max_card*sizeof(int) );
        #pragma omp target update from (result[0:max_card])

        if( save_clusters ){
          for(int i=0; i<max_card; i++){
            debug_out << pnts[2*result[i]] << " " << pnts[2*result[i]+1] << std::endl;
          }
          seeds_out << pnts[2*winner_index] << " " << pnts[2*winner_index+1] << std::endl;
          debug_out.close();
        }
      }

      start = std::chrono::steady_clock::now();

      //update_clustered_pnts_mask<<<grid2D(1), tpb>>>((char *)clustered_pnts_mask, (char *)Ai_mask, max_point_count);
      #pragma omp target teams num_teams(1) thread_limit(tpb)
      {
        #pragma omp parallel 
        {
          int tid = omp_get_thread_num();
          int curThreadCount = omp_get_num_threads();

          // If a point is part of the latest winner cluster, then it should be marked as
          // clustered for the future iterations. Otherwise it should be left as it is.
          for(int i = 0; i+tid < max_point_count; i+=curThreadCount){
            clustered_pnts_mask[i+tid] |= Ai_mask[i+tid];
          }
        }
      }

      end = std::chrono::steady_clock::now();
      update_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      point_count -= max_card;

    }while( max_card > 1 && point_count );

    if( save_clusters ){
      seeds_out.close();
    }

    if( cwrank == 0){
      cout << "QTC is complete. Clustering iteration count: " << iter << std::endl;
      cout << "\nKernel execution time\n";
      cout << "qtc: " << qtc_time * 1e-9f << " (s)\n";
      cout << "trim: " << trim_time * 1e-9f << " (s)\n";
      cout << "update: " << update_time * 1e-9f << " (s)\n";
      cout << "total: " << (qtc_time + trim_time + update_time) * 1e-9f << " (s)\n";
      cout.flush();
    }
  }

  free(dist_source);
  free(indr_mtrx_host);
  free(ungrpd_pnts_indr_host);
  free(cardnl);
  free(result);
  free(degrees);
  free(Ai_mask);
  free(dist_to_clust);
  free(clustered_pnts_mask);

  return;
}
