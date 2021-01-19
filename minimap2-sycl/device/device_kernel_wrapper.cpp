#include <vector>
#include <string>
#include <ctime>
#include <cstdio>
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"
#include "common.h"

score_dt device_ilog2(const score_dt v)
{
    if (v < 2) return 0;
    else if (v < 4) return 1;
    else if (v < 8) return 2;
    else if (v < 16) return 3;
    else if (v < 32) return 4;
    else if (v < 64) return 5;
    else if (v < 128) return 6;
    else if (v < 256) return 7;
    else return 8;
}

score_dt chain_dp_score(const anchor_dt *active, const anchor_dt curr,
        const float avg_qspan, const int max_dist_x, const int max_dist_y, const int bw, const int id)
{
    anchor_dt act;
    *((short4*)&act) = ((short4*)active)[id];

    if (curr.tag != act.tag) return NEG_INF_SCORE_GPU;

    score_dt dist_x = act.x - curr.x;
    if (dist_x == 0 || dist_x > max_dist_x) return NEG_INF_SCORE_GPU;

    score_dt dist_y = act.y - curr.y;
    if (dist_y > max_dist_y || dist_y <= 0) return NEG_INF_SCORE_GPU;

    score_dt dd = dist_x > dist_y ? dist_x - dist_y : dist_y - dist_x;
    if (dd > bw) return NEG_INF_SCORE_GPU;

    score_dt min_d = dist_y < dist_x ? dist_y : dist_x;
    score_dt log_dd = device_ilog2(dd);

    score_dt sc = min_d > act.w ? act.w : min_d;
    sc -= (score_dt)(dd * (0.01 * avg_qspan)) + (log_dd >> 1);

    return sc;
}

void update_anchor(const anchor_dt *in, anchor_dt* out, const int src, const int dst) {
  ((short4*)out)[dst] = ((short4*)in)[src];
}

void update_return(const return_dt *in, return_dt* out, const int src, const int dst) {
  ((short4*)out)[dst] = ((short4*)in)[src];
}

void device_chain_kernel_wrapper(
    std::vector<control_dt> &cont,
    std::vector<anchor_dt> &arg,
    std::vector<return_dt> &ret,
    int max_dist_x, int max_dist_y, int bw)
{
  auto batch_count = cont.size() / PE_NUM;

  control_dt *h_control = (control_dt*) malloc (sizeof(control_dt) * cont.size());
  anchor_dt *h_arg = (anchor_dt*) malloc (arg.size() * sizeof(anchor_dt));
  return_dt *h_ret = (return_dt*) malloc (batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  ret.resize(batch_count * TILE_SIZE * PE_NUM);

  memcpy(h_control, cont.data(), cont.size() * sizeof(control_dt));
  memcpy(h_arg, arg.data(), arg.size() * sizeof(anchor_dt));

  struct timespec start, end;
  clock_gettime(CLOCK_BOOTTIME, &start);

{

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

/*
    buffer<score_dt,2> d_max_tracker (range<2>(PE_NUM, BACK_SEARCH_COUNT_GPU));
    buffer<parent_dt,2> d_j_tracker (range<2>(PE_NUM, BACK_SEARCH_COUNT_GPU));
    d_max_tracker.set_final_data(nullptr);
    d_j_tracker.set_final_data(nullptr);

    int control_size = cont.size();
    int arg_size = arg.size();

    buffer<control_dt, 1> d_control (h_control, cont.size());
    buffer<anchor_dt, 1> d_arg (h_arg, arg.size());
    buffer<return_dt, 1> d_ret (batch_count * TILE_SIZE * PE_NUM);
*/

    range<1> gws (BLOCK_NUM * THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);
    range<1> lws (THREAD_FACTOR * BACK_SEARCH_COUNT_GPU);

    //for (auto batch = 0; batch < batch_count; batch++) {
      q.submit([&] (handler &cgh) {
        //accessor<anchor_dt, 1, sycl_read_write, access::target::local> active_sm(BACK_SEARCH_COUNT_GPU, cgh);
        //accessor<score_dt, 1, sycl_read_write, access::target::local> max_tracker_sm(BACK_SEARCH_COUNT_GPU, cgh);
        //accessor<parent_dt, 1, sycl_read_write, access::target::local> j_tracker_sm(BACK_SEARCH_COUNT_GPU, cgh);

        //auto ret = d_ret.get_access<sycl_write>(cgh); //, range<1>(batch_count*PE_NUM*TILE_SIZE), id<1>(batch * PE_NUM * TILE_SIZE));
        //auto a = d_arg.get_access<sycl_read>(cgh); //, range<1>(arg_size), id<1>(batch * PE_NUM * TILE_SIZE_ACTUAL));
        //auto cont = d_control.get_access<sycl_read>(cgh); //,  range<1>(control_size), id<1>(batch * PE_NUM));
        //auto ret = d_ret.get_access<sycl_write>(cgh, range<1>(PE_NUM*TILE_SIZE), id<1>(batch * PE_NUM * TILE_SIZE));
        //auto a = d_arg.get_access<sycl_read>(cgh, range<1>(PE_NUM * TILE_SIZE_ACTUAL), id<1>(batch * PE_NUM * TILE_SIZE_ACTUAL));
        //auto cont = d_control.get_access<sycl_read>(cgh,  range<1>(PE_NUM), id<1>(batch * PE_NUM));
        //auto max_tracker_g = d_max_tracker.get_access<sycl_read_write>(cgh);
        //auto j_tracker_g = d_j_tracker.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class scan_block>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
/*
          int block = item.get_group(0); 
          int id = item.get_local_id(0);
          int ofs = block;
          auto control = cont[ofs];

          update_anchor(a.get_pointer(), active_sm.get_pointer(), ofs*TILE_SIZE_ACTUAL+id, id);

          if (control.is_new_read) {
            max_tracker_sm[id] = 0;
            j_tracker_sm[id] = -1;
          } else {
            max_tracker_sm[id] = max_tracker_g[ofs][id];
            j_tracker_sm[id] = j_tracker_g[ofs][id];
          }

          for (int i = BACK_SEARCH_COUNT_GPU, curr_idx = 0; curr_idx < TILE_SIZE; i++, curr_idx++) {

            item.barrier(access::fence_space::local_space);
            anchor_dt curr;
            update_anchor(active_sm.get_pointer(), &curr, i % BACK_SEARCH_COUNT_GPU, 0);
            score_dt f_curr = max_tracker_sm[i % BACK_SEARCH_COUNT_GPU];
            parent_dt p_curr = j_tracker_sm[i % BACK_SEARCH_COUNT_GPU];
            if (curr.w >= f_curr) {
              f_curr = curr.w;
              p_curr = (parent_dt)-1;
            }

            // read in new query anchor, put into active array
            item.barrier(access::fence_space::local_space);
            if (id == i % BACK_SEARCH_COUNT_GPU) {
              update_anchor(a.get_pointer(), active_sm.get_pointer(), ofs*TILE_SIZE_ACTUAL+i, id);
              max_tracker_sm[id] = 0;
              j_tracker_sm[id] = -1;
            }

            item.barrier(access::fence_space::local_space);
            score_dt sc = chain_dp_score(active_sm.get_pointer(), curr,
                control.avg_qspan, max_dist_x, max_dist_y, bw, id);

            item.barrier(access::fence_space::local_space);
            if (sc + f_curr >= max_tracker_sm[id]) {
              max_tracker_sm[id] = sc + f_curr;
              j_tracker_sm[id] = (parent_dt)curr_idx + (parent_dt)control.tile_num * TILE_SIZE;
            }

            item.barrier(access::fence_space::local_space);
            if (id == curr_idx % BACK_SEARCH_COUNT_GPU) {
              return_dt tmp;
              tmp.score = f_curr;
              tmp.parent = p_curr;
              update_return(&tmp, ret.get_pointer(), 0, ofs*TILE_SIZE+curr_idx);
            }
          }

          item.barrier(access::fence_space::local_space);
          max_tracker_g[ofs][id] = max_tracker_sm[id];
          j_tracker_g[ofs][id] = j_tracker_sm[id];
*/
        });
      });
    //}
/*
    q.submit([&] (handler &cgh) {
      auto ret = d_ret.get_access<sycl_read>(cgh);
      cgh.copy(ret, h_ret);
    });
    q.wait();
*/
}


  clock_gettime(CLOCK_BOOTTIME, &end);
  printf(" ***** offloading took %f seconds for end-to-end\n",
      ( end.tv_sec - start.tv_sec ) + ( end.tv_nsec - start.tv_nsec ) / 1E9);

  memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  free(h_control);
  free(h_arg);
  free(h_ret);
}

