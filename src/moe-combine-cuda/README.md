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

Each rank maps its expert-output buffer into the other ranks with IPC and
loads it directly in the combine kernel (MORI's P2P-read combine). CUDA and
HIP use runtime IPC handles. SYCL uses
`sycl::ext::oneapi::experimental::ipc::memory`. A device barrier makes every
rank wait until the others have entered the kernel before those loads. MPI
carries the IPC handle bytes and the host barriers. It does not move device
buffers.

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
./main --tokens 128 --hidden 1024 --iters 100 --warmup 100

# one rank per GPU
srun -n 2 ./main --tokens 4096 --hidden 7168 --topk 8
```

`hidden` must be a multiple of 8. `topk` is 1, 2, 4, or 8. Warmup and timed iterations both default to 100. A loaded Cray MPICH module
(`CRAY_MPICH_DIR` or `MPICH_DIR`) selects that prefix and `-lmpi_cray`.
Otherwise the build uses OpenMPI. Override either choice with `MPI_ROOT`
and `MPI_LIB`.
