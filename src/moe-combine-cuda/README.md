# moe-combine

Intra-node Mixture-of-Experts combine, the bandwidth-heavy half of the dispatch
plus combine measured by the [MORI EP benchmark](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-BENCHMARK.md).
One MPI rank owns one GPU. The default shape is the DeepSeek-V3 / MORI bench
point: 4096 tokens, hidden 7168, top-8, 32 experts per rank, bf16 expert outputs.

## What is timed

Routing follows MORI's round-robin benchmark initializer. Expert slot
`token * topk + k` is sent to rank `(token * topk + k) % world`, and each
destination packs arrivals in source-rank order. Combine reads those expert
outputs back and applies the router weights that stayed on the home rank:

```
out[token, h] = sum_k weight[token, k] * expert_output[dest_rank, slot, h]
```

Accumulation is fp32. The result is stored as bf16 and checked against a host
reference.

CUDA and HIP map each rank's expert-output buffer into the other ranks with
IPC and load it directly in the combine kernel (MORI's P2P-read combine). A
device barrier makes every rank wait until the others have entered the kernel
before those loads. This path does not need GPU-aware MPI.

`--transport mpi` (or a failed IPC setup under `--transport auto`) sends the
buffers with `MPI_Isend` / `MPI_Irecv` on device pointers. Before any of those
transfers, ranks 0 and 1 run the same GPU-aware check as `pingpong-*/main-mpi`:
five round trips of a device buffer of 65536 doubles, with rank 1 adding one
on each trip. Rank 0 requires every element to come back as 5. A failure prints
`ERROR: MPI pingpong test failed` and the benchmark stops.

## Bandwidth

Reported the way the MORI C++ bench reports it, in GB/s = bytes / 1e9 / seconds.
The time is the slowest rank.

- **Algo bytes** = `recv_tokens * hidden * sizeof(bf16)`. This counts every
  expert output packed at this rank, including outputs that originated on the
  same rank. MORI's algo bandwidth includes that local traffic.
- **Fabric bytes** = remote reads only. On two ranks this is half of the algo
  bytes, matching MORI's physical cross-node adjustment for a two-node layout
  and, here, the bytes that actually cross XGMI or NVLink.

## Build and run

```bash
# one GPU, no peer traffic
make
./main --tokens 128 --hidden 1024 --iters 2 --warmup 1

# one rank per GPU. Peer IPC is the default.
srun -n 2 ./main --tokens 4096 --hidden 7168 --topk 8 --iters 10 --warmup 5

# GPU-aware MPI instead of IPC. On Cray MPICH, link the matching
# libmpi_gtl_hsa or libmpi_gtl_cuda and export MPICH_GPU_SUPPORT_ENABLED=1.
# The program sets that variable when --transport mpi is selected, if unset.
srun -n 2 ./main --transport mpi --tokens 512 --hidden 7168
```

`hidden` must be a multiple of 8. `topk` is 1, 2, 4, or 8. MPI roots default to
Cray MPICH when that installation is present, otherwise OpenMPI; override
`MPI_ROOT` and `MPI_LIB`.
