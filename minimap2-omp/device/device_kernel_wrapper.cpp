#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "datatypes.h"
#include "kernel_common.h"
#include "memory_scheduler.h"

#pragma omp declare target
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

score_dt chain_dp_score(const anchor_dt *active,
                        const anchor_dt curr,
                        const float avg_qspan,
                        const int max_dist_x,
                        const int max_dist_y,
                        const int bw,
                        const int id)
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
  sc -= (score_dt)(dd * (0.01f * avg_qspan)) + (log_dd >> 1);

  return sc;
}

void update_anchor(const anchor_dt *in, anchor_dt* out, const int src, const int dst) {
  ((short4*)out)[dst] = ((short4*)in)[src];
}

void update_return(const return_dt *in, return_dt* out, const int src, const int dst) {
  ((short4*)out)[dst] = ((short4*)in)[src];
}
#pragma omp end declare target

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

  score_dt max_tracker_g[PE_NUM][BACK_SEARCH_COUNT_GPU];
  parent_dt j_tracker_g[PE_NUM][BACK_SEARCH_COUNT_GPU];

#pragma omp target data map(to: h_control[0:cont.size()], h_arg[0:arg.size()]) \
                        map(alloc: max_tracker_g[0:PE_NUM][0:BACK_SEARCH_COUNT_GPU], \
                                   j_tracker_g[0:PE_NUM][0:BACK_SEARCH_COUNT_GPU]) \
                        map(from: h_ret[0:batch_count * TILE_SIZE * PE_NUM])
  {
    auto k_start = std::chrono::steady_clock::now();

    for (auto batch = 0; batch < batch_count; batch++) {
      #pragma omp target teams num_teams(BLOCK_NUM) thread_limit(BACK_SEARCH_COUNT_GPU)
      {
        anchor_dt active_sm[BACK_SEARCH_COUNT_GPU];
        score_dt max_tracker_sm[BACK_SEARCH_COUNT_GPU];
        parent_dt j_tracker_sm[BACK_SEARCH_COUNT_GPU];
        #pragma omp parallel 
        {
          int block = omp_get_team_num();
          int id = omp_get_thread_num();
          int ofs = block;
          auto control = h_control[batch * PE_NUM + ofs];

          update_anchor(h_arg + batch * PE_NUM * TILE_SIZE_ACTUAL, 
              active_sm, ofs*TILE_SIZE_ACTUAL+id, id);

          if (control.is_new_read) {
            max_tracker_sm[id] = 0;
            j_tracker_sm[id] = -1;
          } else {
            max_tracker_sm[id] = max_tracker_g[ofs][id];
            j_tracker_sm[id] = j_tracker_g[ofs][id];
          }

          for (int i = BACK_SEARCH_COUNT_GPU, curr_idx = 0; curr_idx < TILE_SIZE; i++, curr_idx++) {

            #pragma omp barrier
            anchor_dt curr;
            update_anchor(active_sm, &curr, i % BACK_SEARCH_COUNT_GPU, 0);
            score_dt f_curr = max_tracker_sm[i % BACK_SEARCH_COUNT_GPU];
            parent_dt p_curr = j_tracker_sm[i % BACK_SEARCH_COUNT_GPU];
            if (curr.w >= f_curr) {
              f_curr = curr.w;
              p_curr = (parent_dt)-1;
            }

            // read in new query anchor, put into active array
            #pragma omp barrier
            if (id == i % BACK_SEARCH_COUNT_GPU) {
              update_anchor(h_arg + batch * PE_NUM * TILE_SIZE_ACTUAL,
                  active_sm, ofs*TILE_SIZE_ACTUAL+i, id);
              max_tracker_sm[id] = 0;
              j_tracker_sm[id] = -1;
            }

            #pragma omp barrier
            score_dt sc = chain_dp_score(active_sm, curr,
                control.avg_qspan, max_dist_x, max_dist_y, bw, id);

            #pragma omp barrier
            if (sc + f_curr >= max_tracker_sm[id]) {
              max_tracker_sm[id] = sc + f_curr;
              j_tracker_sm[id] = (parent_dt)curr_idx + (parent_dt)control.tile_num * TILE_SIZE;
            }

            #pragma omp barrier
            if (id == curr_idx % BACK_SEARCH_COUNT_GPU) {
              return_dt tmp;
              tmp.score = f_curr;
              tmp.parent = p_curr;
              update_return(&tmp, h_ret + batch * PE_NUM * TILE_SIZE, 0, ofs*TILE_SIZE+curr_idx);
            }
          }

          #pragma omp barrier
          max_tracker_g[ofs][id] = max_tracker_sm[id];
          j_tracker_g[ofs][id] = j_tracker_sm[id];
        }
      }
    }

    auto k_end = std::chrono::steady_clock::now();
    auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("Total kernel execution time: %f (s)\n", k_time * 1e-9);
  }

  memcpy(ret.data(), h_ret, batch_count * TILE_SIZE * PE_NUM * sizeof(return_dt));

  free(h_control);
  free(h_arg);
  free(h_ret);
}

