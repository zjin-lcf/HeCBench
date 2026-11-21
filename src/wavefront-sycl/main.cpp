// SYCL port of the handbook wavefront kernels.
//
// On AMD the device code calls the same builtins as wavefront-hip. WAVEFRONT_SIZE
// 64 (GFX8/GFX9) uses wave_ror, row_bcast, ballot_w64, and mbcnt_hi. Size 32
// uses row_ror plus ds_bpermute, ds_swizzle, ballot_w32, and mbcnt_lo. Other
// devices use subgroup shuffle, which matches those results. Set WAVEFRONT_SIZE
// to the device wave size. The default Makefile arch is wave64.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef WAVEFRONT_SIZE
#define WAVEFRONT_SIZE 32
#endif

constexpr int kWarp = WAVEFRONT_SIZE;
constexpr int kBlock = 256;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#define WAVEFRONT_AMD 1
#if WAVEFRONT_SIZE == 64
#define WAVEFRONT_CDNA 1
constexpr int dpp_wave_ror1 = 0x13c;
constexpr int dpp_row_bcast15 = 0x142;
constexpr int dpp_row_bcast31 = 0x143;
#else
constexpr int dpp_row_ror1 = 0x121;
constexpr int swizzle_bcast15 = 0x1e0;
#endif
constexpr int dpp_quad_perm_1032 = 0xb1;
constexpr int dpp_quad_perm_2301 = 0x4e;
constexpr int dpp_row_ror4 = 0x124;
constexpr int dpp_row_ror8 = 0x128;
constexpr int swizzle_swap_groups4 = 0x101F;
#endif

static int g_fail = 0;

template <typename T>
static T from_lane(const sycl::sub_group &sg, T val, int src) {
  return sycl::select_from_group(sg, val, src);
}

static int rotate_right(const sycl::sub_group &sg, int val, int lane) {
#ifdef WAVEFRONT_CDNA
  (void)sg;
  (void)lane;
  return __builtin_amdgcn_mov_dpp(val, dpp_wave_ror1, 0xf, 0xf, false);
#elif defined(WAVEFRONT_AMD)
  (void)sg;
  int rotated = __builtin_amdgcn_mov_dpp(val, dpp_row_ror1, 0xf, 0xf, false);
  int addr = (lane == 0) ? 124 : (lane == 16) ? 60 : lane * 4;
  int wrapped = __builtin_amdgcn_ds_bpermute(addr, val);
  return (lane == 0 || lane == 16) ? wrapped : rotated;
#else
  return from_lane(sg, val, (lane + kWarp - 1) % kWarp);
#endif
}

static int swap_groups4(const sycl::sub_group &sg, int val, int lane) {
#ifdef WAVEFRONT_AMD
  (void)sg;
  (void)lane;
  return __builtin_amdgcn_ds_swizzle(val, swizzle_swap_groups4);
#else
  int src = (lane & ~31) | ((lane & 31) ^ 4);
  return from_lane(sg, val, src);
#endif
}

static float shfl_down_reduce(const sycl::sub_group &sg, float val, int lane) {
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    int src = lane + offset;
    if (src >= kWarp)
      src = lane;
    val += from_lane(sg, val, src);
  }
  return val;
}

static float dpp_warp_reduce(const sycl::sub_group &sg, float val, int lane) {
#ifdef WAVEFRONT_AMD
  (void)sg;
  auto as_int = [](float f) { return sycl::bit_cast<int>(f); };
  auto as_float = [](int i) { return sycl::bit_cast<float>(i); };
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_1032, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_quad_perm_2301, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror4, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_ror8, 0xf, 0xf, false));
#ifdef WAVEFRONT_CDNA
  // Full row mask, matching rocPRIM. The sum is in the last lane.
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast15, 0xf, 0xf, false));
  val += as_float(__builtin_amdgcn_mov_dpp(as_int(val), dpp_row_bcast31, 0xf, 0xf, false));
#else
  val += as_float(__builtin_amdgcn_ds_swizzle(as_int(val), swizzle_bcast15));
#endif
  return val;
#else
  if constexpr (kWarp == 32) {
    val += from_lane(sg, val, lane ^ 1);
    val += from_lane(sg, val, lane ^ 2);
    int row = lane & ~15;
    val += from_lane(sg, val, row + ((lane - 4) & 15));
    val += from_lane(sg, val, row + ((lane - 8) & 15));
    val += from_lane(sg, val, 15);
    return val;
  } else {
    for (int offset = kWarp / 2; offset > 0; offset >>= 1)
      val += from_lane(sg, val, lane ^ offset);
    return val;
  }
#endif
}

#if defined(WAVEFRONT_AMD) && defined(__has_builtin)
#if __has_builtin(__builtin_amdgcn_wave_reduce_add_u32)
#define WAVEFRONT_HAS_WAVE_REDUCE 1
#endif
#endif

// wave_reduce is missing from older compilers, so those use a butterfly.
static unsigned wave_sum(const sycl::sub_group &sg, unsigned val, int lane) {
#ifdef WAVEFRONT_HAS_WAVE_REDUCE
  (void)sg;
  (void)lane;
  return __builtin_amdgcn_wave_reduce_add_u32(val, 0);
#else
  for (int offset = kWarp / 2; offset > 0; offset >>= 1)
    val += from_lane(sg, val, lane ^ offset);
  return val;
#endif
}

static int compact_index(const sycl::sub_group &sg, bool keep, int lane) {
#ifdef WAVEFRONT_CDNA
  (void)sg;
  (void)lane;
  unsigned long long mask = __builtin_amdgcn_ballot_w64(keep);
  unsigned lo = static_cast<unsigned>(mask);
  unsigned hi = static_cast<unsigned>(mask >> 32);
  int index = __builtin_amdgcn_mbcnt_lo(lo, 0);
  return __builtin_amdgcn_mbcnt_hi(hi, index);
#elif defined(WAVEFRONT_AMD)
  (void)sg;
  (void)lane;
  unsigned mask = __builtin_amdgcn_ballot_w32(keep);
  return __builtin_amdgcn_mbcnt_lo(mask, 0);
#else
  int pred = keep ? 1 : 0;
  int scan = pred;
  for (int offset = 1; offset < kWarp; offset <<= 1) {
    int src = lane >= offset ? lane - offset : lane;
    int other = from_lane(sg, scan, src);
    if (lane >= offset)
      scan += other;
  }
  return scan - pred;
#endif
}

template <int Src>
static int broadcast_lane(const sycl::sub_group &sg, int val) {
#ifdef WAVEFRONT_AMD
  (void)sg;
  return __builtin_amdgcn_readlane(val, Src);
#else
  return from_lane(sg, val, Src);
#endif
}

static int compiled_path() {
#ifdef WAVEFRONT_CDNA
  return 2;
#elif defined(WAVEFRONT_AMD)
  return 1;
#else
  return 0;
#endif
}

static void report_i(const char *name, double sec, const int *got, const int *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%d ref=%d]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

static void report_f(const char *name, double sec, const float *got, const float *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%g ref=%g]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

static void report_u(const char *name, double sec, const unsigned *got, const unsigned *ref, int n) {
  int mismatch = 0;
  for (int i = 0; i < n; ++i)
    mismatch += got[i] != ref[i];
  if (mismatch == 0) {
    std::printf("PASS %-18s time %.6f s\n", name, sec);
    return;
  }
  g_fail++;
  std::printf("FAIL %-18s mismatch=%d/%d time %.6f s", name, mismatch, n, sec);
  int shown = 0;
  for (int i = 0; i < n && shown < 4; ++i) {
    if (got[i] != ref[i]) {
      std::printf(" [%d got=%u ref=%u]", i, got[i], ref[i]);
      ++shown;
    }
  }
  std::printf("\n");
}

template <typename F>
static double time_launches(sycl::queue &q, int repeat, F &&launch) {
  launch();
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i)
    launch();
  q.wait();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

static void ref_rotate(const std::vector<int> &in, std::vector<int> &out) {
  out = in;
  std::vector<int> nxt(in.size());
  int nwarps = static_cast<int>(in.size()) / kWarp;
  for (int s = 0; s < kWarp; ++s) {
    for (int w = 0; w < nwarps; ++w) {
      for (int lane = 0; lane < kWarp; ++lane) {
        int src = w * kWarp + (lane - 1 + kWarp) % kWarp;
        nxt[w * kWarp + lane] = out[src] ^ s;
      }
    }
    out.swap(nxt);
  }
}

static void ref_swizzle(const std::vector<int> &in, std::vector<int> &out) {
  out.resize(in.size());
  for (int i = 0; i < static_cast<int>(in.size()); ++i) {
    int lane = i % kWarp;
    int src = (lane & ~31) | ((lane & 31) ^ 4);
    out[i] = in[i - lane + src];
  }
}

static void ref_max(const std::vector<int> &in, std::vector<int> &out) {
  out.resize(in.size());
  int nwarps = static_cast<int>(in.size()) / kWarp;
  for (int w = 0; w < nwarps; ++w) {
    int mx = in[w * kWarp];
    for (int lane = 1; lane < kWarp; ++lane)
      mx = std::max(mx, in[w * kWarp + lane]);
    for (int lane = 0; lane < kWarp; ++lane)
      out[w * kWarp + lane] = mx;
  }
}

static void ref_sum_f(const std::vector<float> &in, std::vector<float> &out) {
  int nblocks = static_cast<int>(in.size()) / kBlock;
  out.assign(nblocks, 0.f);
  for (int b = 0; b < nblocks; ++b) {
    float s = 0.f;
    for (int i = 0; i < kBlock; ++i)
      s += in[b * kBlock + i];
    out[b] = s;
  }
}

static void ref_sum_u(const std::vector<unsigned> &in, std::vector<unsigned> &out) {
  int nblocks = static_cast<int>(in.size()) / kBlock;
  out.assign(nblocks, 0u);
  for (int b = 0; b < nblocks; ++b) {
    unsigned s = 0u;
    for (int i = 0; i < kBlock; ++i)
      s += in[b * kBlock + i];
    out[b] = s;
  }
}

static void ref_compact(const std::vector<int> &in, std::vector<int> &out) {
  out.assign(in.size(), 0);
  int nblocks = static_cast<int>(in.size()) / kBlock;
  int warps = kBlock / kWarp;
  for (int b = 0; b < nblocks; ++b) {
    int warp_base = 0;
    for (int w = 0; w < warps; ++w) {
      int index = 0;
      for (int lane = 0; lane < kWarp; ++lane) {
        int gid = b * kBlock + w * kWarp + lane;
        if (in[gid] > 0) {
          out[b * kBlock + warp_base + index] = in[gid];
          ++index;
        }
      }
      warp_base += index;
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::printf("Usage: %s <n> <repeat>\n", argv[0]);
    return 1;
  }
  int n = std::atoi(argv[1]);
  int repeat = std::atoi(argv[2]);
  if (n <= 0 || repeat <= 0) {
    std::printf("FAIL n and repeat must be positive\n");
    return 1;
  }
  n = (n + kBlock - 1) / kBlock * kBlock;

  try {
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
    auto dev_name = q.get_device().get_info<sycl::info::device::name>();

    int *d_info = sycl::malloc_device<int>(2, q);
    q.parallel_for(sycl::nd_range<1>(kBlock, kBlock),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
                     if (it.get_local_id(0) == 0) {
                       d_info[0] = static_cast<int>(it.get_sub_group().get_local_range()[0]);
                       d_info[1] = compiled_path();
                     }
                   })
        .wait();
    int info[2] = {0, 0};
    q.memcpy(info, d_info, sizeof(info)).wait();
    sycl::free(d_info, q);
    int warp = info[0];
    const char *path = info[1] == 2   ? "wave64 wave_ror row_bcast ballot_w64 mbcnt_hi"
                       : info[1] == 1 ? "wave32 ballot_w32 mbcnt_lo row_ror ds_swizzle"
                                      : "subgroup";
    std::printf("INFO device=%s warpSize=%d path=%s n=%d\n", dev_name.c_str(), warp, path, n);
    if (warp != kWarp) {
      std::printf("FAIL warpSize=%d compiled for %d\n", warp, kWarp);
      return 1;
    }

    std::vector<int> hin(n), href(n), got(n);
    std::vector<float> hfin(n), hfref;
    std::vector<unsigned> huin(n), huref;
    for (int i = 0; i < n; ++i) {
      hin[i] = ((i * 17 + 3) % 31) - 10;
      hfin[i] = static_cast<float>((i % 13) + 1);
      huin[i] = static_cast<unsigned>((i % 13) + 1);
    }

    int nblocks = n / kBlock;
    int *d_in = sycl::malloc_device<int>(n, q);
    int *d_out = sycl::malloc_device<int>(n, q);
    float *d_fin = sycl::malloc_device<float>(n, q);
    float *d_fout = sycl::malloc_device<float>(nblocks, q);
    unsigned *d_uin = sycl::malloc_device<unsigned>(n, q);
    unsigned *d_uout = sycl::malloc_device<unsigned>(nblocks, q);
    q.memcpy(d_in, hin.data(), n * sizeof(int));
    q.memcpy(d_fin, hfin.data(), n * sizeof(float));
    q.memcpy(d_uin, huin.data(), n * sizeof(unsigned)).wait();

    sycl::nd_range<1> ndr(sycl::range<1>(n), sycl::range<1>(kBlock));
    sycl::nd_range<1> ndr_blocks(sycl::range<1>(n), sycl::range<1>(kBlock));

    auto run_i = [&](const char *name, auto submit, auto ref) {
      ref(hin, href);
      double sec = time_launches(q, repeat, submit);
      q.memcpy(got.data(), d_out, n * sizeof(int)).wait();
      report_i(name, sec, got.data(), href.data(), n);
    };

    run_i(
        "readlane",
        [&] {
          q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            auto sg = it.get_sub_group();
            int lane = static_cast<int>(sg.get_local_linear_id());
            int gid = static_cast<int>(it.get_global_id(0));
            int val = (gid < n) ? d_in[gid] : 0;
            int max_val = val;
            for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
              int src = lane + offset;
              if (src >= kWarp)
                src = lane;
              max_val = std::max(max_val, from_lane(sg, max_val, src));
            }
            int broadcast = broadcast_lane<0>(sg, max_val);
            if (gid < n)
              d_out[gid] = broadcast;
          });
        },
        ref_max);

    run_i(
        "rotate-dpp",
        [&] {
          q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            auto sg = it.get_sub_group();
            int lane = static_cast<int>(sg.get_local_linear_id());
            int gid = static_cast<int>(it.get_global_id(0));
            int val = (gid < n) ? d_in[gid] : 0;
            for (int i = 0; i < kWarp; ++i) {
              val = rotate_right(sg, val, lane);
              val ^= i;
            }
            if (gid < n)
              d_out[gid] = val;
          });
        },
        ref_rotate);

    run_i(
        "rotate-shfl",
        [&] {
          q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            auto sg = it.get_sub_group();
            int lane = static_cast<int>(sg.get_local_linear_id());
            int gid = static_cast<int>(it.get_global_id(0));
            int val = (gid < n) ? d_in[gid] : 0;
            for (int i = 0; i < kWarp; ++i) {
              val = from_lane(sg, val, (lane + kWarp - 1) % kWarp);
              val ^= i;
            }
            if (gid < n)
              d_out[gid] = val;
          });
        },
        ref_rotate);

    run_i(
        "swizzle",
        [&] {
          q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
            auto sg = it.get_sub_group();
            int lane = static_cast<int>(sg.get_local_linear_id());
            int gid = static_cast<int>(it.get_global_id(0));
            int val = (gid < n) ? d_in[gid] : 0;
            if (gid < n)
              d_out[gid] = swap_groups4(sg, val, lane);
          });
        },
        ref_swizzle);

    std::vector<float> fgot(nblocks);
    ref_sum_f(hfin, hfref);
    double sec = time_launches(q, repeat, [&] {
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(kBlock / 32), h);
        h.parallel_for(ndr_blocks, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
          auto sg = it.get_sub_group();
          int lane = static_cast<int>(sg.get_local_linear_id());
          int warpid = static_cast<int>(it.get_local_id(0)) / kWarp;
          int gid = static_cast<int>(it.get_global_id(0));
          float val = (gid < n) ? d_fin[gid] : 0.f;
          val = shfl_down_reduce(sg, val, lane);
          if (lane == 0)
            sdata[warpid] = val;
          it.barrier(sycl::access::fence_space::local_space);
          if (warpid == 0) {
            int num_warps = kBlock / kWarp;
            val = (lane < num_warps) ? sdata[lane] : 0.f;
            val = shfl_down_reduce(sg, val, lane);
          }
          if (it.get_local_id(0) == 0)
            d_fout[it.get_group(0)] = val;
        });
      });
    });
    q.memcpy(fgot.data(), d_fout, nblocks * sizeof(float)).wait();
    report_f("reduce-shfl", sec, fgot.data(), hfref.data(), nblocks);

    sec = time_launches(q, repeat, [&] {
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> sdata(sycl::range<1>(kBlock / 32), h);
        h.parallel_for(ndr_blocks, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
          auto sg = it.get_sub_group();
          int lane = static_cast<int>(sg.get_local_linear_id());
          int warpid = static_cast<int>(it.get_local_id(0)) / kWarp;
          int gid = static_cast<int>(it.get_global_id(0));
          float val = (gid < n) ? d_fin[gid] : 0.f;
          val = dpp_warp_reduce(sg, val, lane);
          if (lane == kWarp - 1)
            sdata[warpid] = val;
          it.barrier(sycl::access::fence_space::local_space);
          if (warpid == 0) {
            int num_warps = kBlock / kWarp;
            val = (lane < num_warps) ? sdata[lane] : 0.f;
            val = dpp_warp_reduce(sg, val, lane);
          }
          if (static_cast<int>(it.get_local_id(0)) == kWarp - 1)
            d_fout[it.get_group(0)] = val;
        });
      });
    });
    q.memcpy(fgot.data(), d_fout, nblocks * sizeof(float)).wait();
    report_f("reduce-dpp", sec, fgot.data(), hfref.data(), nblocks);

    std::vector<unsigned> ugot(nblocks);
    ref_sum_u(huin, huref);
    sec = time_launches(q, repeat, [&] {
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<unsigned, 1> sdata(sycl::range<1>(kBlock / 32), h);
        h.parallel_for(ndr_blocks, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
          auto sg = it.get_sub_group();
          int lane = static_cast<int>(sg.get_local_linear_id());
          int warpid = static_cast<int>(it.get_local_id(0)) / kWarp;
          int gid = static_cast<int>(it.get_global_id(0));
          unsigned val = (gid < n) ? d_uin[gid] : 0u;
          unsigned sum = wave_sum(sg, val, lane);
          if (lane == 0)
            sdata[warpid] = sum;
          it.barrier(sycl::access::fence_space::local_space);
          if (warpid == 0) {
            int num_warps = kBlock / kWarp;
            sum = (lane < num_warps) ? sdata[lane] : 0u;
            sum = wave_sum(sg, sum, lane);
          }
          if (it.get_local_id(0) == 0)
            d_uout[it.get_group(0)] = sum;
        });
      });
    });
    q.memcpy(ugot.data(), d_uout, nblocks * sizeof(unsigned)).wait();
    report_u("reduce-wave", sec, ugot.data(), huref.data(), nblocks);

    ref_compact(hin, href);
    q.memset(d_out, 0, n * sizeof(int)).wait();
    sec = time_launches(q, repeat, [&] {
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<int, 1> sdata(sycl::range<1>(kBlock / 32), h);
        h.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kWarp)]] {
          auto sg = it.get_sub_group();
          int lane = static_cast<int>(sg.get_local_linear_id());
          int warpid = static_cast<int>(it.get_local_id(0)) / kWarp;
          int gid = static_cast<int>(it.get_global_id(0));
          int val = (gid < n) ? d_in[gid] : 0;
          bool keep = (gid < n) && (val > 0);
          int index = compact_index(sg, keep, lane);
          int warp_count = broadcast_lane<kWarp - 1>(sg, index + (keep ? 1 : 0));
          if (lane == 0)
            sdata[warpid] = warp_count;
          it.barrier(sycl::access::fence_space::local_space);
          int warp_base = 0;
          for (int w = 0; w < warpid; ++w)
            warp_base += sdata[w];
          if (keep)
            d_out[it.get_group(0) * kBlock + warp_base + index] = val;
        });
      });
    });
    q.memcpy(got.data(), d_out, n * sizeof(int)).wait();
    report_i("compact", sec, got.data(), href.data(), n);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
    sycl::free(d_fin, q);
    sycl::free(d_fout, q);
    sycl::free(d_uin, q);
    sycl::free(d_uout, q);
  } catch (const sycl::exception &e) {
    std::printf("FAIL sycl %s\n", e.what());
    return 1;
  }

  if (g_fail) {
    std::printf("FAIL kernels=%d\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
