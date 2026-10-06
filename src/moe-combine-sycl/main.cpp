// SYCL MoE combine. One MPI rank owns one GPU.
// Multi-GPU traffic uses GPU-aware MPI on device allocations, so ranks 0 and 1
// first run the pingpong benchmark's device-buffer check. The peer-IPC path
// used by the CUDA and HIP variants is not portable across SYCL backends.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include "reference.hpp"

static int integer_arg(const char *text, const char *name, long minimum, long maximum) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < minimum || value > maximum) {
    std::fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

static std::uint16_t dev_f32_to_bf16(float x) {
  const std::uint32_t bits0 = sycl::bit_cast<std::uint32_t>(x);
  const std::uint32_t lsb = (bits0 >> 16) & 1u;
  return static_cast<std::uint16_t>((bits0 + 0x7fffu + lsb) >> 16);
}

static float dev_bf16_to_f32(std::uint16_t b) {
  return sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

// pingpong-*/main-mpi: five device round trips, rank 1 adds one, rank 0 expects 5.
static int gpu_aware_mpi_check(sycl::queue &q, int rank, int world) {
  if (world < 2)
    return 1;
  const long n = 1L << 16;
  int ok = 1;
  if (rank < 2) {
    double *d_a = sycl::malloc_device<double>(n, q);
    if (!d_a)
      return 0;
    q.memset(d_a, 0, sizeof(double) * n).wait();
    const int tag1 = 10;
    const int tag2 = 20;
    for (int i = 1; i <= 5; ++i) {
      if (rank == 0) {
        MPI_Send(d_a, n, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_a, n, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(d_a, n, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        q.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)),
                       [=](sycl::id<1> id) { d_a[id] += 1.0; })
            .wait();
        MPI_Send(d_a, n, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    if (rank == 0) {
      std::vector<double> host(n);
      q.memcpy(host.data(), d_a, sizeof(double) * n).wait();
      for (long i = 0; i < n; ++i) {
        if (host[i] != 5.0) {
          std::printf("ERROR: MPI pingpong test failed\n");
          ok = 0;
          break;
        }
      }
    }
    sycl::free(d_a, q);
  }
  int all_ok = 0;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  return all_ok;
}

static void exchange(int rank, int world, int hidden, const std::vector<int> &recv,
                     std::uint16_t *stage, const std::vector<std::uint16_t *> &copies) {
  std::vector<MPI_Request> reqs;
  reqs.reserve(static_cast<std::size_t>(world - 1) * 2);
  for (int peer = 0; peer < world; ++peer) {
    if (peer == rank)
      continue;
    const std::size_t send_bytes =
        static_cast<std::size_t>(recv[rank]) * hidden * sizeof(std::uint16_t);
    const std::size_t recv_bytes =
        static_cast<std::size_t>(recv[peer]) * hidden * sizeof(std::uint16_t);
    if (send_bytes > static_cast<std::size_t>(INT_MAX) ||
        recv_bytes > static_cast<std::size_t>(INT_MAX)) {
      std::fprintf(stderr, "exchange is larger than MPI can send in one message\n");
      std::exit(EXIT_FAILURE);
    }
    MPI_Request send_req, recv_req;
    MPI_Irecv(copies[peer], static_cast<int>(recv_bytes), MPI_BYTE, peer, 42,
              MPI_COMM_WORLD, &recv_req);
    MPI_Isend(stage, static_cast<int>(send_bytes), MPI_BYTE, peer, 42, MPI_COMM_WORLD,
              &send_req);
    reqs.push_back(recv_req);
    reqs.push_back(send_req);
  }
  if (!reqs.empty())
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
}

template <int kTopk>
static void launch_combine(sycl::queue &q, int blocks, int tokens, int hidden,
                           const int *dest, const int *slot,
                           const std::uint16_t *const *stage, std::uint16_t *out) {
  q.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(blocks) * 256, 256),
                 [=](sycl::nd_item<1> item) {
                   const int nvec = hidden >> 3;
                   const int thread = item.get_local_id(0);
                   for (int token = item.get_group(0); token < tokens; token += blocks) {
                     for (int vec = thread; vec < nvec; vec += 256) {
                       float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                       const int *td = dest + static_cast<std::size_t>(token) * kTopk;
                       const int *ts = slot + static_cast<std::size_t>(token) * kTopk;
                       for (int k = 0; k < kTopk; ++k) {
                         const std::uint16_t *src =
                             stage[td[k]] + (static_cast<std::size_t>(ts[k]) * hidden +
                                             (static_cast<std::size_t>(vec) << 3));
                         const std::uint32_t *w =
                             reinterpret_cast<const std::uint32_t *>(src);
                         float x[8];
#pragma unroll
                         for (int i = 0; i < 4; ++i) {
                           x[2 * i] = dev_bf16_to_f32(static_cast<std::uint16_t>(w[i]));
                           x[2 * i + 1] =
                               dev_bf16_to_f32(static_cast<std::uint16_t>(w[i] >> 16));
                         }
                         const float weight = weight_of(token, k, kTopk);
#pragma unroll
                         for (int j = 0; j < 8; ++j)
                           acc[j] += weight * x[j];
                       }
                       std::uint16_t *dst = out + static_cast<std::size_t>(token) * hidden +
                                            (static_cast<std::size_t>(vec) << 3);
                       std::uint32_t packed[4];
#pragma unroll
                       for (int i = 0; i < 4; ++i) {
                         const unsigned lo = dev_f32_to_bf16(acc[2 * i]);
                         const unsigned hi = dev_f32_to_bf16(acc[2 * i + 1]);
                         packed[i] = lo | (hi << 16);
                       }
                       *reinterpret_cast<sycl::vec<std::uint32_t, 4> *>(dst) =
                           sycl::vec<std::uint32_t, 4>(packed[0], packed[1], packed[2],
                                                       packed[3]);
                     }
                   }
                 });
}

int main(int argc, char **argv) {
  int tokens = 4096;
  int hidden = 7168;
  int topk = 8;
  int experts = 32;
  int iterations = 10;
  int warmup = 5;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (has_value && !std::strcmp(argv[i], "--tokens"))
      tokens = integer_arg(argv[++i], "token count", 1, 1 << 20);
    else if (has_value && !std::strcmp(argv[i], "--hidden"))
      hidden = integer_arg(argv[++i], "hidden size", 8, 16384);
    else if (has_value && !std::strcmp(argv[i], "--topk"))
      topk = integer_arg(argv[++i], "top-k", 1, 8);
    else if (has_value && !std::strcmp(argv[i], "--experts"))
      experts = integer_arg(argv[++i], "experts per rank", 1, 4096);
    else if (has_value && !std::strcmp(argv[i], "--iters"))
      iterations = integer_arg(argv[++i], "iteration count", 1, 100000);
    else if (has_value && !std::strcmp(argv[i], "--warmup"))
      warmup = integer_arg(argv[++i], "warmup count", 0, 100000);
    else {
      std::fprintf(stderr,
                   "usage: %s [--tokens N] [--hidden N] [--topk 1|2|4|8] "
                   "[--experts N] [--iters N] [--warmup N]\n",
                   argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (hidden % 8 != 0 || (topk != 1 && topk != 2 && topk != 4 && topk != 8)) {
    std::fprintf(stderr, "hidden size must be a multiple of 8 and topk must be 1, 2, 4, or 8\n");
    return EXIT_FAILURE;
  }

  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  try {
    const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) {
      if (rank == 0)
        std::fprintf(stderr, "no GPU was found\n");
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    sycl::queue q{gpus[rank % static_cast<int>(gpus.size())],
                  sycl::property::queue::in_order()};
    const std::string name = q.get_device().get_info<sycl::info::device::name>();

    const CombineRoute route = build_route(world, tokens, topk, hidden, experts);
    const int slots = route.recv[rank];
    const std::size_t stage_n =
        std::max(static_cast<std::size_t>(slots) * hidden, std::size_t{1});
    const std::size_t out_n = static_cast<std::size_t>(tokens) * hidden;
    const std::size_t map_n = static_cast<std::size_t>(tokens) * topk;

    std::uint16_t *stage = sycl::malloc_device<std::uint16_t>(stage_n, q);
    std::uint16_t *out = sycl::malloc_device<std::uint16_t>(out_n, q);
    int *dest = sycl::malloc_device<int>(map_n, q);
    int *slot = sycl::malloc_device<int>(map_n, q);
    if (!stage || !out || !dest || !slot)
      throw std::bad_alloc();
    const std::size_t base = route.index(rank, 0, 0);
    q.memcpy(dest, route.dest.data() + base, sizeof(int) * map_n);
    q.memcpy(slot, route.slot.data() + base, sizeof(int) * map_n);
    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(slots) * hidden), [=](sycl::id<1> id) {
          const int index = static_cast<int>(id);
          const int local_slot = index / hidden;
          const int h = index - local_slot * hidden;
          stage[id] = dev_f32_to_bf16(stage_unit(rank, local_slot, h));
        })
        .wait();

    if (world > 1 && !gpu_aware_mpi_check(q, rank, world)) {
      MPI_Finalize();
      return EXIT_FAILURE;
    }

    std::vector<std::uint16_t *> stage_ptrs(world, nullptr);
    std::vector<std::uint16_t *> copies(world, nullptr);
    stage_ptrs[rank] = stage;
    for (int peer = 0; peer < world; ++peer) {
      if (peer == rank)
        continue;
      const std::size_t n =
          std::max(static_cast<std::size_t>(route.recv[peer]) * hidden, std::size_t{1});
      copies[peer] = sycl::malloc_device<std::uint16_t>(n, q);
      if (!copies[peer])
        throw std::bad_alloc();
      stage_ptrs[peer] = copies[peer];
    }
    auto **d_stage = sycl::malloc_device<std::uint16_t *>(world, q);
    q.memcpy(d_stage, stage_ptrs.data(), sizeof(std::uint16_t *) * world).wait();
    MPI_Barrier(MPI_COMM_WORLD);

    const int blocks = tokens < 256 ? tokens : 256;
    auto step = [&] {
      if (world > 1)
        exchange(rank, world, hidden, route.recv, stage, copies);
      if (topk == 1)
        launch_combine<1>(q, blocks, tokens, hidden, dest, slot, d_stage, out);
      else if (topk == 2)
        launch_combine<2>(q, blocks, tokens, hidden, dest, slot, d_stage, out);
      else if (topk == 4)
        launch_combine<4>(q, blocks, tokens, hidden, dest, slot, d_stage, out);
      else
        launch_combine<8>(q, blocks, tokens, hidden, dest, slot, d_stage, out);
      q.wait_and_throw();
    };
    for (int i = 0; i < warmup; ++i)
      step();
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
      step();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
    std::vector<std::uint16_t> host(out_n);
    q.memcpy(host.data(), out, out_n * sizeof(std::uint16_t)).wait();
    const double local_error = max_abs_error(route, rank, host.data(), 8);
    const int pass_bit = std::isfinite(local_error) && local_error < 2.0e-2;
    double max_ms = 0;
    double max_error = 0;
    int all_pass = 0;
    MPI_Reduce(&ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&pass_bit, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
      const double seconds = max_ms / 1.0e3;
      const double algo_bytes =
          static_cast<double>(route.recv[0]) * hidden * sizeof(std::uint16_t);
      const double fabric_bytes =
          static_cast<double>(remote_slots(route, 0)) * hidden * sizeof(std::uint16_t);
      std::printf("device: %s\n", name.c_str());
      std::printf("ranks: %d  tokens: %d  hidden: %d  topk: %d  experts_per_rank: %d\n",
                  world, tokens, hidden, topk, experts);
      std::printf("transport: %s\n", world > 1 ? "mpi" : "local");
      std::printf("recv_tokens: %d  iterations: %d  warmup: %d\n", route.recv[0], iterations,
                  warmup);
      std::printf("kernel: %.6f ms\n", max_ms);
      std::printf("algo_bandwidth: %.3f GB/s\n", algo_bytes / 1.0e9 / seconds);
      std::printf("fabric_bandwidth: %.3f GB/s\n", fabric_bytes / 1.0e9 / seconds);
      std::printf("max_abs_error: %.3e\n", max_error);
      std::printf("moe-combine: %s\n", all_pass ? "PASS" : "FAIL");
    }

    sycl::free(d_stage, q);
    for (std::uint16_t *p : copies)
      if (p)
        sycl::free(p, q);
    sycl::free(stage, q);
    sycl::free(out, q);
    sycl::free(dest, q);
    sycl::free(slot, q);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception &ex) {
    std::fprintf(stderr, "SYCL error: %s\n", ex.what());
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    return EXIT_FAILURE;
  }
}
