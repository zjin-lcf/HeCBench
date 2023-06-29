#include "kernel_common.h"
#include "datatypes.h"

using short4 = sycl::short4;

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

score_dt chain_dp_score(
  const anchor_dt *active,
  const anchor_dt curr,
  const float avg_qspan,
  const int max_dist_x,
  const int max_dist_y,
  const int bw, const int id)
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

void device_chain_tiled(
  return_dt *__restrict ret,
  const anchor_dt *__restrict a,
  const control_dt *__restrict cont,
  score_dt *__restrict max_tracker_g,
  parent_dt *__restrict j_tracker_g,
  anchor_dt *__restrict active,
  score_dt *__restrict max_tracker,
  parent_dt *__restrict j_tracker,
  sycl::nd_item<1> &item,
  const int max_dist_x,
  const int max_dist_y,
  const int bw)
{
  int block = item.get_group(0);
  int id = item.get_local_id(0);
  int ofs = block;
  auto control = cont[ofs];

  ((short4*)active)[id] = ((short4*)a)[ofs * TILE_SIZE_ACTUAL + id];
  if (control.is_new_read) {
    max_tracker[id] = 0;
    j_tracker[id] = -1;
  } else {
    max_tracker[id] = max_tracker_g[ofs * BACK_SEARCH_COUNT_GPU + id];
    j_tracker[id] = j_tracker_g[ofs * BACK_SEARCH_COUNT_GPU + id];
  }

  for (int i = BACK_SEARCH_COUNT_GPU, curr_idx = 0; curr_idx < TILE_SIZE; i++, curr_idx++) {

    item.barrier(sycl::access::fence_space::local_space);
    anchor_dt curr;
    *((short4*)&curr) = ((short4*)active)[i % BACK_SEARCH_COUNT_GPU];
    score_dt f_curr = max_tracker[i % BACK_SEARCH_COUNT_GPU];
    parent_dt p_curr = j_tracker[i % BACK_SEARCH_COUNT_GPU];
    if (curr.w >= f_curr) {
      f_curr = curr.w;
      p_curr = (parent_dt)-1;
    }

    /* read in new query anchor, put into active array*/
    item.barrier(sycl::access::fence_space::local_space);
    if (id == i % BACK_SEARCH_COUNT_GPU) {
      ((short4*)active)[id] = ((short4*)a)[ofs * TILE_SIZE_ACTUAL + i];
      max_tracker[id] = 0;
      j_tracker[id] = -1;
    }

    item.barrier(sycl::access::fence_space::local_space);
    score_dt sc = chain_dp_score(active, curr,
        control.avg_qspan, max_dist_x, max_dist_y, bw, id);

    item.barrier(sycl::access::fence_space::local_space);
    if (sc + f_curr >= max_tracker[id]) {
      max_tracker[id] = sc + f_curr;
      j_tracker[id] = (parent_dt)curr_idx + (parent_dt)control.tile_num * TILE_SIZE;
    }

    item.barrier(sycl::access::fence_space::local_space);
    if (id == curr_idx % BACK_SEARCH_COUNT_GPU) {
      return_dt tmp;
      tmp.score = f_curr;
      tmp.parent = p_curr;
      ((short4*)ret)[ofs * TILE_SIZE + curr_idx] = *((short4*)&tmp);
    }
  }

  item.barrier(sycl::access::fence_space::local_space);
  max_tracker_g[ofs * BACK_SEARCH_COUNT_GPU + id] = max_tracker[id];
  j_tracker_g[ofs * BACK_SEARCH_COUNT_GPU + id] = j_tracker[id];
}
