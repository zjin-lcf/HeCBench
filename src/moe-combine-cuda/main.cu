// Multi-GPU MoE combine. One MPI rank owns one GPU.
//
// The default transport maps each rank's expert-output buffer into the others
// with CUDA/HIP IPC and reads it from the combine kernel. That is the
// intra-node path measured by the MORI EP benchmark, and it does not need
// GPU-aware MPI.
//
// `--transport mpi`, or a failed IPC setup in `--transport auto`, moves those
// buffers with MPI_Isend/MPI_Irecv on device pointers. Before any such
// transfer, ranks 0 and 1 run the pingpong benchmark's GPU-aware MPI check
// (five device-buffer round trips, the receiver adding one each time).

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <mpi.h>
#include "combine.cuh"

#if defined(__HIPCC__)
#define MC_CHECK(call)                                                         \
  do {                                                                         \
    hipError_t err_ = (call);                                                  \
    if (err_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,        \
                   hipGetErrorString(err_));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
using mc_ipc_t = hipIpcMemHandle_t;
static constexpr unsigned mc_ipc_flag = hipIpcMemLazyEnablePeerAccess;
#define mcMalloc hipMalloc
#define mcFree hipFree
#define mcMemcpy hipMemcpy
#define mcMemset hipMemset
#define mcDeviceSynchronize hipDeviceSynchronize
#define mcGetDeviceCount hipGetDeviceCount
#define mcSetDevice hipSetDevice
#define mcGetDeviceProperties hipGetDeviceProperties
#define mcGetLastError hipGetLastError
#define mcIpcGetMemHandle hipIpcGetMemHandle
#define mcIpcOpenMemHandle hipIpcOpenMemHandle
#define mcIpcCloseMemHandle hipIpcCloseMemHandle
#define mcMemcpyHostToDevice hipMemcpyHostToDevice
#define mcMemcpyDeviceToHost hipMemcpyDeviceToHost
using mc_prop_t = hipDeviceProp_t;
#else
#define MC_CHECK(call)                                                         \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err_));                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
using mc_ipc_t = cudaIpcMemHandle_t;
static constexpr unsigned mc_ipc_flag = cudaIpcMemLazyEnablePeerAccess;
#define mcMalloc cudaMalloc
#define mcFree cudaFree
#define mcMemcpy cudaMemcpy
#define mcMemset cudaMemset
#define mcDeviceSynchronize cudaDeviceSynchronize
#define mcGetDeviceCount cudaGetDeviceCount
#define mcSetDevice cudaSetDevice
#define mcGetDeviceProperties cudaGetDeviceProperties
#define mcGetLastError cudaGetLastError
#define mcIpcGetMemHandle cudaIpcGetMemHandle
#define mcIpcOpenMemHandle cudaIpcOpenMemHandle
#define mcIpcCloseMemHandle cudaIpcCloseMemHandle
#define mcMemcpyHostToDevice cudaMemcpyHostToDevice
#define mcMemcpyDeviceToHost cudaMemcpyDeviceToHost
using mc_prop_t = cudaDeviceProp;
#endif

enum class Transport { Auto, Peer, Mpi };

static int integer_arg(const char *text, const char *name, long minimum,
                       long maximum) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || *end || value < minimum || value > maximum) {
    std::fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

// pingpong-*/main-mpi: five round trips of a device buffer. Rank 1 adds one
// on each trip, so rank 0 must observe 5.
static int gpu_aware_mpi_check(int rank, int world) {
  if (world < 2)
    return 1;
  const long n = 1L << 16;
  int ok = 1;
  if (rank < 2) {
    double *d_a = nullptr;
    MC_CHECK(mcMalloc(&d_a, sizeof(double) * n));
    MC_CHECK(mcMemset(d_a, 0, sizeof(double) * n));
    MC_CHECK(mcDeviceSynchronize());
    const int tag1 = 10;
    const int tag2 = 20;
    for (int i = 1; i <= 5; ++i) {
      if (rank == 0) {
        MPI_Send(d_a, n, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_a, n, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(d_a, n, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mpi_check_add<<<1024, 256>>>(d_a, n);
        MC_CHECK(mcGetLastError());
        MC_CHECK(mcDeviceSynchronize());
        MPI_Send(d_a, n, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    if (rank == 0) {
      std::vector<double> host(n);
      MC_CHECK(mcMemcpy(host.data(), d_a, sizeof(double) * n, mcMemcpyDeviceToHost));
      for (long i = 0; i < n; ++i) {
        if (host[i] != 5.0) {
          std::printf("ERROR: MPI pingpong test failed\n");
          ok = 0;
          break;
        }
      }
    }
    MC_CHECK(mcFree(d_a));
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
    MPI_Request send_req, recv_req;
    const std::size_t send_bytes =
        static_cast<std::size_t>(recv[rank]) * hidden * sizeof(std::uint16_t);
    const std::size_t recv_bytes =
        static_cast<std::size_t>(recv[peer]) * hidden * sizeof(std::uint16_t);
    if (send_bytes > static_cast<std::size_t>(INT_MAX) ||
        recv_bytes > static_cast<std::size_t>(INT_MAX)) {
      std::fprintf(stderr, "exchange is larger than MPI can send in one message\n");
      std::exit(EXIT_FAILURE);
    }
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

static void launch_combine(bool peer, int topk, int blocks, int tokens, int hidden,
                           int world, int rank, const int *dest, const int *slot,
                           const std::uint16_t *const *stage, std::uint16_t *out,
                           unsigned *arrival, unsigned long long *local_flags,
                           unsigned long long *const *peer_flags,
                           unsigned long long epoch) {
  switch (topk) {
  case 1:
    if (peer)
      combine_kernel<1, true><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                              slot, stage, out, arrival, local_flags,
                                              peer_flags, epoch);
    else
      combine_kernel<1, false><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                               slot, stage, out, arrival, local_flags,
                                               peer_flags, epoch);
    break;
  case 2:
    if (peer)
      combine_kernel<2, true><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                              slot, stage, out, arrival, local_flags,
                                              peer_flags, epoch);
    else
      combine_kernel<2, false><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                               slot, stage, out, arrival, local_flags,
                                               peer_flags, epoch);
    break;
  case 4:
    if (peer)
      combine_kernel<4, true><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                              slot, stage, out, arrival, local_flags,
                                              peer_flags, epoch);
    else
      combine_kernel<4, false><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                               slot, stage, out, arrival, local_flags,
                                               peer_flags, epoch);
    break;
  case 8:
    if (peer)
      combine_kernel<8, true><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                              slot, stage, out, arrival, local_flags,
                                              peer_flags, epoch);
    else
      combine_kernel<8, false><<<blocks, 256>>>(tokens, hidden, world, rank, dest,
                                               slot, stage, out, arrival, local_flags,
                                               peer_flags, epoch);
    break;
  default:
    std::fprintf(stderr, "topk must be 1, 2, 4, or 8\n");
    std::exit(EXIT_FAILURE);
  }
  MC_CHECK(mcGetLastError());
}

int main(int argc, char **argv) {
  int tokens = 4096;
  int hidden = 7168;
  int topk = 8;
  int experts = 32;
  int iterations = 10;
  int warmup = 5;
  Transport transport = Transport::Auto;
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
    else if (has_value && !std::strcmp(argv[i], "--transport")) {
      ++i;
      if (!std::strcmp(argv[i], "auto"))
        transport = Transport::Auto;
      else if (!std::strcmp(argv[i], "peer"))
        transport = Transport::Peer;
      else if (!std::strcmp(argv[i], "mpi"))
        transport = Transport::Mpi;
      else {
        std::fprintf(stderr, "invalid transport: %s\n", argv[i]);
        return EXIT_FAILURE;
      }
    } else {
      std::fprintf(stderr,
                   "usage: %s [--tokens N] [--hidden N] [--topk 1|2|4|8] "
                   "[--experts N] [--iters N] [--warmup N] "
                   "[--transport auto|peer|mpi]\n",
                   argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (hidden % 8 != 0) {
    std::fprintf(stderr, "hidden size must be a multiple of 8\n");
    return EXIT_FAILURE;
  }
  if (topk != 1 && topk != 2 && topk != 4 && topk != 8) {
    std::fprintf(stderr, "topk must be 1, 2, 4, or 8\n");
    return EXIT_FAILURE;
  }

  // Cray MPICH uses this to send device buffers. It has to be set before
  // MPI_Init, and only the explicit MPI transport needs it.
  if (transport == Transport::Mpi)
    setenv("MPICH_GPU_SUPPORT_ENABLED", "1", 0);
  MPI_Init(&argc, &argv);
  int rank = 0;
  int world = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (world > 256) {
    if (rank == 0)
      std::fprintf(stderr, "at most 256 MPI ranks are supported\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int devices = 0;
  MC_CHECK(mcGetDeviceCount(&devices));
  if (devices < 1) {
    if (rank == 0)
      std::fprintf(stderr, "no GPU was found\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  MC_CHECK(mcSetDevice(rank % devices));
  mc_prop_t prop{};
  MC_CHECK(mcGetDeviceProperties(&prop, rank % devices));

  const CombineRoute route = build_route(world, tokens, topk, hidden, experts);
  const int slots = route.recv[rank];
  const std::size_t stage_bytes =
      static_cast<std::size_t>(slots) * hidden * sizeof(std::uint16_t);
  const std::size_t stage_alloc = std::max(stage_bytes, sizeof(std::uint16_t));
  const std::size_t out_bytes =
      static_cast<std::size_t>(tokens) * hidden * sizeof(std::uint16_t);
  const std::size_t map_n = static_cast<std::size_t>(tokens) * topk;

  std::uint16_t *stage = nullptr;
  std::uint16_t *out = nullptr;
  unsigned long long *flags = nullptr;
  unsigned *arrival = nullptr;
  int *dest = nullptr;
  int *slot = nullptr;
  MC_CHECK(mcMalloc(&stage, stage_alloc));
  MC_CHECK(mcMalloc(&out, out_bytes));
#if defined(__HIPCC__)
  if (hipExtMallocWithFlags(reinterpret_cast<void **>(&flags),
                            sizeof(unsigned long long) * world,
                            hipDeviceMallocFinegrained) != hipSuccess)
#endif
    MC_CHECK(mcMalloc(&flags, sizeof(unsigned long long) * world));
  MC_CHECK(mcMalloc(&arrival, sizeof(unsigned) * 2));
  MC_CHECK(mcMalloc(&dest, sizeof(int) * map_n));
  MC_CHECK(mcMalloc(&slot, sizeof(int) * map_n));
  MC_CHECK(mcMemset(flags, 0, sizeof(unsigned long long) * world));
  MC_CHECK(mcMemset(arrival, 0, sizeof(unsigned) * 2));
  const std::size_t base = route.index(rank, 0, 0);
  MC_CHECK(mcMemcpy(dest, route.dest.data() + base, sizeof(int) * map_n,
                    mcMemcpyHostToDevice));
  MC_CHECK(mcMemcpy(slot, route.slot.data() + base, sizeof(int) * map_n,
                    mcMemcpyHostToDevice));

  std::vector<std::uint16_t *> stage_ptrs(world, nullptr);
  std::vector<unsigned long long *> flag_ptrs(world, nullptr);
  std::vector<void *> opened;
  bool peer = transport != Transport::Mpi && world > 1;
  if (world == 1)
    peer = true;

  if (peer && world > 1) {
    mc_ipc_t stage_handle{};
    mc_ipc_t flag_handle{};
    int local_ok = mcIpcGetMemHandle(&stage_handle, stage) ==
#if defined(__HIPCC__)
                       hipSuccess
#else
                       cudaSuccess
#endif
                   && mcIpcGetMemHandle(&flag_handle, flags) ==
#if defined(__HIPCC__)
                          hipSuccess;
#else
                          cudaSuccess;
#endif
    std::vector<int> oks(world, 0);
    MPI_Allgather(&local_ok, 1, MPI_INT, oks.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<mc_ipc_t> stage_handles(world), flag_handles(world);
    MPI_Allgather(&stage_handle, sizeof(mc_ipc_t), MPI_BYTE, stage_handles.data(),
                  sizeof(mc_ipc_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&flag_handle, sizeof(mc_ipc_t), MPI_BYTE, flag_handles.data(),
                  sizeof(mc_ipc_t), MPI_BYTE, MPI_COMM_WORLD);
    bool all_ok = true;
    for (int ok : oks)
      all_ok = all_ok && ok;
    if (all_ok) {
      stage_ptrs[rank] = stage;
      flag_ptrs[rank] = flags;
      for (int p = 0; p < world && all_ok; ++p) {
        if (p == rank)
          continue;
        void *sp = nullptr;
        void *fp = nullptr;
        if (mcIpcOpenMemHandle(&sp, stage_handles[p], mc_ipc_flag) ||
            mcIpcOpenMemHandle(&fp, flag_handles[p], mc_ipc_flag)) {
          all_ok = false;
          break;
        }
        stage_ptrs[p] = static_cast<std::uint16_t *>(sp);
        flag_ptrs[p] = static_cast<unsigned long long *>(fp);
        opened.push_back(sp);
        opened.push_back(fp);
      }
    }
    int opened_ok = all_ok ? 1 : 0;
    int opened_all = 0;
    MPI_Allreduce(&opened_ok, &opened_all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!opened_all) {
      for (void *p : opened)
        (void)mcIpcCloseMemHandle(p);
      opened.clear();
      peer = false;
      if (transport == Transport::Peer) {
        if (rank == 0)
          std::fprintf(stderr, "peer transport is not available between these GPUs\n");
        MPI_Finalize();
        return EXIT_FAILURE;
      }
    }
  } else {
    stage_ptrs[rank] = stage;
    flag_ptrs[rank] = flags;
  }

  std::vector<std::uint16_t *> copies(world, nullptr);
  if (!peer && world > 1) {
    if (!gpu_aware_mpi_check(rank, world)) {
      MPI_Finalize();
      return EXIT_FAILURE;
    }
    for (int p = 0; p < world; ++p) {
      if (p == rank) {
        stage_ptrs[p] = stage;
        continue;
      }
      const std::size_t bytes = std::max(
          static_cast<std::size_t>(route.recv[p]) * hidden * sizeof(std::uint16_t),
          sizeof(std::uint16_t));
      MC_CHECK(mcMalloc(&copies[p], bytes));
      stage_ptrs[p] = copies[p];
    }
    flag_ptrs[rank] = flags;
  }

  std::uint16_t **d_stage_ptrs = nullptr;
  unsigned long long **d_flag_ptrs = nullptr;
  MC_CHECK(mcMalloc(&d_stage_ptrs, sizeof(std::uint16_t *) * world));
  MC_CHECK(mcMalloc(&d_flag_ptrs, sizeof(unsigned long long *) * world));
  MC_CHECK(mcMemcpy(d_stage_ptrs, stage_ptrs.data(), sizeof(std::uint16_t *) * world,
                    mcMemcpyHostToDevice));
  MC_CHECK(mcMemcpy(d_flag_ptrs, flag_ptrs.data(), sizeof(unsigned long long *) * world,
                    mcMemcpyHostToDevice));

  const int fill_blocks = 256;
  fill_stage<<<fill_blocks, 256>>>(stage, rank, slots, hidden);
  MC_CHECK(mcGetLastError());
  MC_CHECK(mcDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  const int blocks = tokens < 256 ? tokens : 256;
  unsigned long long epoch = 1;
  // One kernel at a time. A newer launch stores a newer epoch into the same
  // flag slots, and an in-flight kernel waiting on the older epoch would hang.
  auto step = [&] {
    if (!peer && world > 1)
      exchange(rank, world, hidden, route.recv, stage, copies);
    launch_combine(peer && world > 1, topk, blocks, tokens, hidden, world, rank, dest,
                   slot, d_stage_ptrs, out, arrival, flags, d_flag_ptrs, epoch);
    ++epoch;
    MC_CHECK(mcDeviceSynchronize());
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
  std::vector<std::uint16_t> host(static_cast<std::size_t>(tokens) * hidden);
  MC_CHECK(mcMemcpy(host.data(), out, out_bytes, mcMemcpyDeviceToHost));
  const double local_error = max_abs_error(route, rank, host.data(), 8);
  const bool local_pass = std::isfinite(local_error) && local_error < 2.0e-2;

  double max_ms = 0;
  double max_error = 0;
  int all_pass = 0;
  const int pass_bit = local_pass ? 1 : 0;
  MPI_Reduce(&ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Allreduce(&pass_bit, &all_pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (rank == 0) {
    const double seconds = max_ms / 1.0e3;
    const double algo_bytes =
        static_cast<double>(route.recv[0]) * hidden * sizeof(std::uint16_t);
    const double fabric_bytes =
        static_cast<double>(remote_slots(route, 0)) * hidden * sizeof(std::uint16_t);
    std::printf("device: %s\n", prop.name);
    std::printf("ranks: %d  tokens: %d  hidden: %d  topk: %d  experts_per_rank: %d\n",
                world, tokens, hidden, topk, experts);
    std::printf("transport: %s\n", (peer && world > 1) || world == 1 ? "peer" : "mpi");
    std::printf("recv_tokens: %d  iterations: %d  warmup: %d\n", route.recv[0],
                iterations, warmup);
    std::printf("kernel: %.6f ms\n", max_ms);
    std::printf("algo_bandwidth: %.3f GB/s\n", algo_bytes / 1.0e9 / seconds);
    std::printf("fabric_bandwidth: %.3f GB/s\n", fabric_bytes / 1.0e9 / seconds);
    std::printf("max_abs_error: %.3e\n", max_error);
    std::printf("moe-combine: %s\n", all_pass ? "PASS" : "FAIL");
  }

  MC_CHECK(mcFree(d_stage_ptrs));
  MC_CHECK(mcFree(d_flag_ptrs));
  for (void *p : opened)
    MC_CHECK(mcIpcCloseMemHandle(p));
  for (std::uint16_t *p : copies)
    if (p)
      MC_CHECK(mcFree(p));
  MC_CHECK(mcFree(stage));
  MC_CHECK(mcFree(out));
  MC_CHECK(mcFree(flags));
  MC_CHECK(mcFree(arrival));
  MC_CHECK(mcFree(dest));
  MC_CHECK(mcFree(slot));
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
