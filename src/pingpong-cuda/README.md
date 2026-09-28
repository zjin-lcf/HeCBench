# pingpong-cuda

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-nccl` does not.

Use exactly two MPI ranks (one GPU per rank). Run `./main-mpi` only for the
tests below; `make run` also launches `main-nccl`.

Rebuild against the MPI you launch with. Mixing an NVHPC-linked binary with
distro `mpirun` (or the reverse) is not a valid test.

## GPU-aware MPI test (must pass)

Link and launch with the CUDA-aware MPI from the NVIDIA HPC SDK (Makefile
`MPI_ROOT` / `LAUNCHER`):

```bash
make clean
make ARCH=sm_90
# Use the same MPI the binary was linked against, e.g. Makefile LAUNCHER:
/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/comm_libs/mpi/bin/mpirun --mca coll ^hcoll -n 2 ./main-mpi
```

**Pass:** no error on stderr; rank 0 prints
`MPI : Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

Startup asks the MPI library with `MPIX_Query_cuda_support()` when that
function is in the MPI headers (Open MPI `mpi-ext.h`, or MPICH 4.0.1 and
later). If the library reports no CUDA buffer support, the program aborts
before any device pointer is passed to MPI. If the library reports support,
or the headers have no query, one 512 KiB device transfer is checked. A
watchdog aborts that transfer if it is still running after 30 s.

## Non-GPU-aware MPI test (must fail quickly)

Rebuild against a **host-only** MPI, then run **that** `mpirun`:

```bash
make clean
make ARCH=sm_90 MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the process exits instead of blocking in `MPI_Recv`.
Typical stderr lines:

- `MPI library reports no CUDA GPU-buffer support` — the library query returned no
- `GPU-aware MPI probe failed` — the query was missing or said yes, and the device buffer was wrong
- `GPU-aware MPI probe timed out` — MPI hung on a device pointer; a watchdog aborts within about 30 s
