# pingpong-hip

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-nccl` (RCCL) does not.

Use exactly two MPI ranks (one GPU per rank). Run `./main-mpi` only for the
tests below; `make run` also launches `main-nccl`.

The Makefile default `MPI_ROOT` (`/usr/lib/x86_64-linux-gnu/openmpi`) is often
**host-only**. Rebuild against GPU-aware MPI for the pass test (ROCm-aware
OpenMPI, or Cray MPICH plus `libmpi_gtl_hsa`). Launch with the same MPI.

## GPU-aware MPI test (must pass)

```bash
# Cray MPICH on AMD GPUs (link -lmpi_gtl_hsa)
export MPICH_GPU_SUPPORT_ENABLED=1
srun -n 2 ./main-mpi
```

```bash
# ROCm-aware OpenMPI (not the typical distro package)
make clean
make MPI_ROOT=/path/to/rocm-aware-openmpi
/path/to/rocm-aware-openmpi/bin/mpirun -n 2 ./main-mpi
```

**Pass:** no error on stderr; rank 0 prints
`MPI : Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

Startup asks the MPI library with `MPIX_Query_hip_support()` (MPICH 4.0.1
and later) or `MPIX_Query_rocm_support()` (Open MPI 5, from `mpi-ext.h`).
Open MPI 4 has no ROCm query. If the library reports no HIP buffer support,
the program aborts before any device pointer is passed to MPI. If the
library reports support, or the headers have no query, one 512 KiB device
transfer is checked. A watchdog aborts that transfer if it is still running
after 30 s. `MPICH_GPU_SUPPORT_ENABLED=0` can still reach that transfer:
Cray MPICH may report HIP support while the variable has GPU support turned
off.

## Non-GPU-aware MPI test (must fail quickly)

Same `main-mpi` sources, GPU support off or a host-only MPI **after a rebuild**
against that MPI:

```bash
# Cray MPICH: GPU support off (same GPU-aware-linked binary)
export MPICH_GPU_SUPPORT_ENABLED=0
srun -n 2 ./main-mpi
```

```bash
# Distro / host-only OpenMPI (Makefile default)
make clean
make MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the process exits instead of blocking in `MPI_Recv`.
Typical stderr lines:

- `MPI library reports no HIP GPU-buffer support` — the library query returned no
- `GPU-aware MPI probe failed` — the query was missing or said yes, and the device buffer was wrong
- `GPU-aware MPI probe timed out` — MPI hung on a device pointer; a watchdog aborts within about 30 s
