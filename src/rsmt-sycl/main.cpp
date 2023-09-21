#include <stdio.h>
#include <sys/time.h>
#include "common.h"
#include "utils.h"

template <int WarpsPerBlock, int PinLimit>
static 
void buildMST(const ID num, 
              const ctype* const __restrict x,
              const ctype* const __restrict y,
              edge* const __restrict edges,
              ctype dist[PinLimit],
              nd_item<1> &item)
{
  auto g = item.get_group();

  multi_ptr<ID[WarpsPerBlock][PinLimit], access::address_space::local_space> 
    t1 = ext::oneapi::group_local_memory_for_overwrite<ID[WarpsPerBlock][PinLimit]>(g);
  ID (*source)[PinLimit] = *t1;

  multi_ptr<ID[WarpsPerBlock][PinLimit], access::address_space::local_space> 
    t2 = ext::oneapi::group_local_memory_for_overwrite<ID[WarpsPerBlock][PinLimit]>(g);
  ID (*destin)[PinLimit] = *t2;

  multi_ptr<ctype[WarpsPerBlock], access::address_space::local_space>
    t3 = ext::oneapi::group_local_memory_for_overwrite<ctype[WarpsPerBlock]>(g);
  ctype *mindj = *t3;

  const int lane = item.get_local_id(0) % WS;
  const int warp = item.get_local_id(0) / WS;

  // initialize
  ID numItems = num - 1;
  for (ID i = lane; i < numItems; i += WS) dist[i] = INT_MAX;  // change if ctype changed
  for (ID i = lane; i < numItems; i += WS) destin[warp][i] = (ID)(i + 1);

  // Prim's MST algorithm
  ID src = 0;
  auto sg = item.get_sub_group();

  for (ID cnt = 0; cnt < num - 1; cnt++) {
    sycl::group_barrier(sg);
    if (lane == 0) mindj[warp] = INT_MAX;

    // update distances
    sycl::group_barrier(sg);
    for (ID j = lane; j < numItems; j += WS) {
      const ID dst = destin[warp][j];
      const ctype dnew =
          sycl::abs(x[src] - x[dst]) + sycl::abs(y[src] - y[dst]);
      ctype d = dist[j];
      if (d > dnew) {
        d = dnew;
        dist[j] = dnew;
        source[warp][j] = src;
      }
      const int upv = d * (MaxPins * 2) + j;  // tie breaker for determinism
      auto ao = ext::oneapi::atomic_ref<ctype,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::work_group,
                access::address_space::local_space> (mindj[warp]);
      ao.fetch_min(upv);
    }

    // create new edge
    sycl::group_barrier(sg);
    const ID j = mindj[warp] % (MaxPins * 2);
    src = destin[warp][j];
    numItems--;
    if (lane == 0) {
      edges[cnt].src = source[warp][j];
      edges[cnt].dst = src;
      dist[j] = dist[numItems];
      source[warp][j] = source[warp][numItems];
      destin[warp][j] = destin[warp][numItems];
    }
  }
}

template <int WarpsPerBlock, int PinLimit>
static 
bool insertSteinerPoints(ID& num,
                         ctype* const __restrict x,
                         ctype* const __restrict y,
                         const edge* const __restrict edges,
                         ctype dist[PinLimit],
                         nd_item<1> &item)
{
  auto g = item.get_group();

  multi_ptr<ID[WarpsPerBlock][PinLimit][8], access::address_space::local_space> 
    t1 = ext::oneapi::group_local_memory_for_overwrite<ID[WarpsPerBlock][PinLimit][8]>(g);
  ID (*adj)[PinLimit][8] = *t1;

  multi_ptr<int[WarpsPerBlock][PinLimit], access::address_space::local_space> 
    t2 = ext::oneapi::group_local_memory_for_overwrite<int[WarpsPerBlock][PinLimit]>(g);
  int (*cnt)[PinLimit] = *t2;

  const int lane = item.get_local_id(0) % WS;
  const int warp = item.get_local_id(0) / WS;
  const ID top = num;

  // create adjacency lists
  auto sg = item.get_sub_group();

  for (ID i = lane; i < top; i += WS) cnt[warp][i] = 0;

  sycl::group_barrier(sg);

  for (ID e = lane; e < top - 1; e += WS) {
    dist[e] = -1;
    const ID s = edges[e].src;
    const ID d = edges[e].dst;
    if ((x[d] != x[s]) || (y[d] != y[s])) {

      auto as = ext::oneapi::atomic_ref<int,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::work_group,
                access::address_space::local_space> (cnt[warp][s]);

      const int ps = as.fetch_add(1);

      adj[warp][s][ps] = e;

      auto ad = ext::oneapi::atomic_ref<int,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::work_group,
                access::address_space::local_space> (cnt[warp][d]);

      const int pd = ad.fetch_add(1);

      adj[warp][d][pd] = e;
    }
  }

  // find best distance for each triangle
  sycl::group_barrier(sg);
  for (ID s = lane; s < top; s += WS) {
    if (cnt[warp][s] >= 2) {
      const ctype x0 = x[s];
      const ctype y0 = y[s];
      for (char j = 0; j < cnt[warp][s] - 1; j++) {
        const ID e1 = adj[warp][s][j];
        const ID d1 = (s != edges[e1].src) ? edges[e1].src : edges[e1].dst;
        const ctype x1 = x[d1];
        const ctype y1 = y[d1];
        for (char k = j + 1; k < cnt[warp][s]; k++) {
          const ID e2 = adj[warp][s][k];
          const ID d2 = (s != edges[e2].src) ? edges[e2].src : edges[e2].dst;
          const ctype stx =
              sycl::max(sycl::min(x0, x1), sycl::min(sycl::max(x0, x1), x[d2]));
          const ctype sty =
              sycl::max(sycl::min(y0, y1), sycl::min(sycl::max(y0, y1), y[d2]));
          const ctype rd = sycl::abs(stx - x0) + sycl::abs(sty - y0);
          if (rd > 0) {
            const ctype rd1 = rd * (MaxPins * 2) + e1;  // tie breaker
            const ctype rd2 = rd * (MaxPins * 2) + e2;  // tie breaker

            auto a1 = ext::oneapi::atomic_ref<ctype,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::work_group,
                access::address_space::local_space> (dist[e1]);
            a1.fetch_max(rd2);


            auto a2 = ext::oneapi::atomic_ref<ctype,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::work_group,
                access::address_space::local_space> (dist[e2]);
            a2.fetch_max(rd1);
          }
        }
      }
    }
  }

  // process "triangles" to find best candidate Steiner points
  sycl::group_barrier(sg);

  bool updated = false;
  for (ID e1 = lane; sycl::any_of_group(sg, e1 < top - 2); e1 += WS) {
    bool insert = false;
    ctype stx, sty;
    if (e1 < top - 2) {
      const ctype d1 = dist[e1];
      if (d1 > 0) {
        const ID e2 = d1 % (MaxPins * 2);
        if (e2 > e1) {
          const ctype d2 = dist[e2];
          if (e1 == d2 % (MaxPins * 2)) {
            const ctype x0 = x[edges[e1].src];
            const ctype y0 = y[edges[e1].src];
            const ctype x1 = x[edges[e1].dst];
            const ctype y1 = y[edges[e1].dst];
            ctype x2 = x[edges[e2].src];
            ctype y2 = y[edges[e2].src];
            if (((x2 == x0) && (y2 == y0)) || ((x2 == x1) && (y2 == y1))) {
              x2 = x[edges[e2].dst];
              y2 = y[edges[e2].dst];
            }
            updated = true;
            insert = true;
            stx = sycl::max(sycl::min(x0, x1), sycl::min(sycl::max(x0, x1), x2));
            sty = sycl::max(sycl::min(y0, y1), sycl::min(sycl::max(y0, y1), y2));
          }
        }
      }
    }
    const int bal = sycl::reduce_over_group(
        sg, insert ? (0x1 << sg.get_local_linear_id()) : 0,
        sycl::ext::oneapi::plus<>());
    const int pos = sycl::popcount(bal & ~(-1 << lane)) + num;
    if (insert) {
      x[pos] = stx;
      y[pos] = sty;
    }
    num += sycl::popcount(bal);
  }

  return sycl::any_of_group(sg, updated);
}

template <int WarpsPerBlock, int PinLimit>
static 
inline void processSmallNet(const int i,
                            const int* const __restrict idxin,
                            const ctype* const __restrict xin,
                            const ctype* const __restrict yin,
                            int* const __restrict idxout,
                            ctype* const __restrict xout,
                            ctype* const __restrict yout,
                             edge* const __restrict edges,
                              int* const __restrict wl,
                              nd_item<1> &item,
                              int* const __restrict wlsize)
{
  auto g = item.get_group();

  multi_ptr<ctype[WarpsPerBlock][PinLimit], access::address_space::local_space> 
    t1 = ext::oneapi::group_local_memory_for_overwrite<ctype[WarpsPerBlock][PinLimit]>(g);
  ctype (*dist)[PinLimit] = *t1;

  const int lane = item.get_local_id(0) % WS;
  const int warp = item.get_local_id(0) / WS;

  // initialize arrays and copy input coords to output
  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;
  if (lane == 0) idxout[i] = pout;
  for (ID j = lane; j < num; j += WS) xout[pout + j] = xin[pin + j];
  for (ID j = lane; j < num; j += WS) yout[pout + j] = yin[pin + j];

  // process nets
  if (num == 2) {
    if (lane == 0) edges[pout] = edge{0, 1};
  } else if (num == 3) {
    ctype x0, y0;
    if (lane < 3) {
      edges[pout + lane] = edge{(short)lane, 3};
      x0 = xout[pout + lane];
      y0 = yout[pout + lane];
    }
    auto sg = item.get_sub_group();
    const ctype x1 = select_from_group(sg, x0, 1);
    const ctype y1 = select_from_group(sg, y0, 1);
    const ctype x2 = select_from_group(sg, x0, 2);
    const ctype y2 = select_from_group(sg, y0, 2);
    if (lane == 0) {
      xout[pout + 3] =
          sycl::max(sycl::min(x0, x1), sycl::min(sycl::max(x0, x1), x2));
      yout[pout + 3] =
          sycl::max(sycl::min(y0, y1), sycl::min(sycl::max(y0, y1), y2));
    }
  } else if (num <= 32) {
    // iterate until all Steiner points added
    ID cnt = num;
    do {
      buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout],
                                        &edges[pout], dist[warp], item);
    } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(
        cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp], item));
  } else {
    if (lane == 0) {
      auto ao = ext::oneapi::atomic_ref<int,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::device,
                access::address_space::global_space> (*wlsize);
      wl[ao.fetch_add(1)] = i;
    }
  }
}

template <int WarpsPerBlock, int PinLimit>
static 
inline void processLargeNet(const int i,
                            const int* const __restrict idxin,
                            ctype* const __restrict xout,
                            ctype* const __restrict yout,
                             edge* const __restrict edges,
                             nd_item<1> &item)
{
  auto g = item.get_group();

  multi_ptr<ctype[WarpsPerBlock][PinLimit], access::address_space::local_space> 
    t1 = ext::oneapi::group_local_memory_for_overwrite<ctype[WarpsPerBlock][PinLimit]>(g);
  ctype (*dist)[PinLimit] = *t1;

  const int warp = item.get_local_id(0) / WS;

  const int pin = idxin[i];
  const ID num = idxin[i + 1] - pin;
  const int pout = 2 * pin;

  // iterate until all Steiner points added
  ID cnt = num;
  do {
    buildMST<WarpsPerBlock, PinLimit>(cnt, &xout[pout], &yout[pout],
                                      &edges[pout], dist[warp], item);
  } while (insertSteinerPoints<WarpsPerBlock, PinLimit>(
      cnt, &xout[pout], &yout[pout], &edges[pout], dist[warp], item));
}

template <int WarpsPerBlock, int PinLimit>
static 
void largeNetKernel(const int* const __restrict idxin,
                    const ctype* const __restrict xin,
                    const ctype* const __restrict yin,
                    int* const __restrict idxout,
                    ctype* __restrict xout,
                    ctype* __restrict yout,
                     edge* __restrict edges,
                    const int numnets,
                    int* const __restrict wl,
                    nd_item<1> &item,
                    int *__restrict currpos1,
                    int *__restrict wlsize)
{
  // compute Steiner points and edges
  const int lane = item.get_local_id(0) % WS;
  auto sg = item.get_sub_group();

  do {
    int i;
    if (lane == 0) {
      auto ao = ext::oneapi::atomic_ref<int,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::device,
                access::address_space::global_space> (*currpos1);
      i = ao.fetch_add(1);
    }
    i = select_from_group(sg, i, 0);
    if (i >= numnets) break;
    processSmallNet<WarpsPerBlock, PinLimit>(
        i, idxin, xin, yin, idxout, xout, yout, edges, wl, item, wlsize);
  } while (true);

  // set final element
  if ((item.get_local_id(0) == 0) && (item.get_group(0) == 0)) {
    idxout[numnets] = 2 * idxin[numnets];
  }
}

template <int WarpsPerBlock, int PinLimit>
static 
void smallNetKernel(const int* const __restrict idxin,
                    ctype* __restrict xout,
                    ctype* __restrict yout,
                     edge* __restrict edges,
                      int* const __restrict wl,
                      nd_item<1> &item,
                      int* const __restrict currpos2,
                      int* const __restrict wlsize)
{
  // compute Steiner points and edges
  const int lane = item.get_local_id(0) % WS;
  auto sg = item.get_sub_group();

  do {
    int i;
    if (lane == 0) {
      auto ao = ext::oneapi::atomic_ref<int,
                ext::oneapi::memory_order::relaxed,
                ext::oneapi::memory_scope::device,
                access::address_space::global_space> (*currpos2);
      i = ao.fetch_add(1);
    }

    i = select_from_group(sg, i, 0);
    if (i >= *wlsize) break;
    processLargeNet<WarpsPerBlock, PinLimit>(wl[i], idxin, xout, yout, edges, item);
  } while (true);
}

static void computeRSMT(queue &q,
                        const int* const __restrict idxin,
                        const ctype* const __restrict xin,
                        const ctype* const __restrict yin,
                         int* const __restrict idxout,
                       ctype* const __restrict xout,
                       ctype* const __restrict yout,
                        edge* const __restrict edges,
                        const int numnets)
{
  // obtain GPU info
  const int SMs = q.get_device().get_info<info::device::max_compute_units>();
  const int blocks = SMs * 2;
  printf("launching %d thread blocks with %d threads per block\n", blocks, 24 * WS);

  // allocate and initialize GPU memory
  int* d_idxin;  ctype* d_xin;  ctype* d_yin;
  int* d_idxout;  ctype* d_xout;  ctype* d_yout;  edge* d_edges;
  int* d_wl;
  const int size = idxin[numnets];

  int *d_currpos1, *d_currpos2, *d_wlsize;
  d_currpos1 = sycl::malloc_device<int>(1, q);
  q.memset(d_currpos1, 0, sizeof(int));

  d_currpos2 = sycl::malloc_device<int>(1, q);
  q.memset(d_currpos2, 0, sizeof(int));

  d_wlsize = sycl::malloc_device<int>(1, q);
  q.memset(d_wlsize, 0, sizeof(int));

  d_idxin = sycl::malloc_device<int>((numnets + 1), q);
  d_xin = sycl::malloc_device<ctype>(size, q);
  d_yin = sycl::malloc_device<ctype>(size, q);
  d_idxout = sycl::malloc_device<int>((numnets + 1), q);
  d_xout = sycl::malloc_device<ctype>(2 * size, q);
  d_yout = sycl::malloc_device<ctype>(2 * size, q);
  d_edges = sycl::malloc_device<edge>(2 * size, q);
  d_wl = sycl::malloc_device<int>(numnets, q);
  q.memcpy(d_idxin, idxin, (numnets + 1) * sizeof(int));
  q.memcpy(d_xin, xin, size * sizeof(ctype));
  q.memcpy(d_yin, yin, size * sizeof(ctype));

  q.wait();

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // process nets
  q.memset(d_xout, -1, 2 * size * sizeof(ctype));
  q.memset(d_yout, -1, 2 * size * sizeof(ctype));
  q.memset(d_edges, 0, 2 * size * sizeof(edge));

  range<1> gws_l (24 * blocks * WS);
  range<1> lws_l (24 * WS);

  q.submit([&](handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws_l, lws_l), [=](nd_item<1> item)
       [[intel::reqd_sub_group_size(32)]] {
          largeNetKernel<24, 64>(
              d_idxin, d_xin, d_yin, d_idxout, d_xout, d_yout, d_edges, numnets,
              d_wl, item, d_currpos1, d_wlsize);
    });
  });

  range<1> gws_s (3 * blocks * WS);
  range<1> lws_s (3 * WS);

  q.submit([&](handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws_s, lws_s), [=](nd_item<1> item)
      [[intel::reqd_sub_group_size(32)]] {
          smallNetKernel<3, 512>(d_idxin, d_xout, d_yout, d_edges, d_wl,
                                 item, d_currpos2, d_wlsize);
    });
  });

  // end time
  q.wait();
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);
  printf("throughput: %.f nets/sec\n", numnets / runtime);

  // transfer results from GPU
  q.memcpy(idxout, d_idxout, (numnets + 1) * sizeof(int));
  q.memcpy(xout, d_xout, 2 * size * sizeof(ctype));
  q.memcpy(yout, d_yout, 2 * size * sizeof(ctype));
  q.memcpy(edges, d_edges, 2 * size * sizeof(edge));
  q.wait();

  // clean up
  free(d_currpos1, q);
  free(d_currpos2, q);
  free(d_wlsize, q);
  free(d_wl, q);
  free(d_edges, q);
  free(d_yout, q);
  free(d_xout, q);
  free(d_idxout, q);
  free(d_yin, q);
  free(d_xin, q);
  free(d_idxin, q);
}

int main(int argc, char* argv[])
{
  printf("A Simple, Fast, and GPU-friendly Steiner-Tree Heuristic\n");
  printf("Copyright 2019-2022 Texas State University\n\n");

  if (argc != 2) {
    printf("Usage: %s file_name \n", argv[0]);
    exit(-1);
  }

  printf("reading input file: %s\n", argv[1]);
  grid g;
  net_list n;
  read_file(argv[1], g, n);
  const int numnets = n.num_net;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  int* idxin = NULL;
  idxin = malloc_host<int>((numnets + 1), q);

  // initialize idxin
  idxin[0] = 0;
  ID hipin = 0;
  int pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    hipin = std::max(hipin, num);
    pos += num;
    idxin[i + 1] = pos;
  }

  // histogram and trucated nets
  int trunc = 0;
  int* const hist = new int [hipin + 1];
  for (int i = 0; i < hipin + 1; i++) hist[i] = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    if (num < n.net_num_pins[i]) trunc++;
    hist[num]++;
  }
  int sum = 0;
  for (int i = 0; i < hipin + 1; i++) {
    sum += hist[i];
  }
  delete [] hist;

  printf("number of nets: %d\n", numnets);
  printf("max pins per net: %d\n", hipin);
  printf("truncated nets: %d\n", trunc);
  if (hipin > MaxPins) {printf("ERROR: hi_pin_count must be no more than %d\n", MaxPins); exit(-1);}

  // pin coordinates
  ctype* xin = malloc_host<ctype>(idxin[numnets], q);
  ctype* yin = malloc_host<ctype>(idxin[numnets], q);

  // initialize pin coordinates
  pos = 0;
  for (int i = 0; i < numnets; i++) {
    const ID num = idxin[i + 1] - idxin[i];
    for (ID j = 0; j < num; j++) xin[pos + j] = std::get<0>(n.num_net_arr[i][j]);
    for (ID j = 0; j < num; j++) yin[pos + j] = std::get<1>(n.num_net_arr[i][j]);
    pos += num;
  }

  // result storage
  const int size = 2 * idxin[numnets];
  int* idxout = malloc_host<int>((numnets + 1), q);

  ctype* xout = malloc_host<ctype>(size, q);

  ctype* yout = malloc_host<ctype>(size, q);

  edge* edges = malloc_host<edge>(size, q);

  // compute Steiner points and edges
  computeRSMT(q, idxin, xin, yin, idxout, xout, yout, edges, numnets);

  // print results
  long total_len = 0, total_pin = 0;
  
  for (int i = 0; i < numnets; i++) {
    // body of treeLength function illustrates how to read solution
    const ctype len = treeLength(idxout[i + 1] - idxout[i],
                      &xout[idxout[i]], &yout[idxout[i]], &edges[idxout[i]]);
    const ID num = std::min(n.net_num_pins[i], MaxPins);
    total_len += len;
    total_pin += num;
  }
  printf("total wirelength: %ld\n", total_len);
  printf("total pins: %ld\n", total_pin);

  // clean up
  free_memory(g, n);
  free(edges, q);
  free(yout, q);
  free(xout, q);
  free(idxout, q);
  free(yin, q);
  free(xin, q);
  free(idxin, q);

  return 0;
}
