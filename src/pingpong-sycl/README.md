# pingpong-sycl

GPU ping-pong bandwidth. `main-mpi` passes **device pointers** to
`MPI_Send` / `MPI_Recv` (GPU-aware MPI required). `main-ccl` does not.

Use exactly two MPI ranks (one GPU per rank). Run `./main-mpi` only for the
tests below; `make run` also launches `main-ccl`.

Match the SYCL backend, the MPI you **link**, and the launcher. Do not mix
`make HIP=yes` with the Makefile default Intel `MPI_ROOT`.

## GPU-aware MPI test (must pass)

HIP backend and Cray MPICH (link `libmpi_gtl_hsa`):

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export ONEAPI_DEVICE_SELECTOR=hip:gpu
srun -n 2 ./main-mpi
```

Intel GPU / Level Zero and Intel MPI (Makefile `MPI_ROOT`).
`I_MPI_OFFLOAD` defaults to 0, which turns GPU buffers off:

```bash
make clean
make
export I_MPI_OFFLOAD=1
/opt/intel/oneapi/2024.1/bin/mpirun -n 2 ./main-mpi
```

NVIDIA GPU and CUDA-aware MPI (`CUDA=yes`, NVHPC MPI — see Makefile comments):

```bash
make clean
make CUDA=yes CUDA_ARCH=sm_90
# launch with the NVHPC mpirun used at link time
```

**Pass:** no error on stderr; rank 0 prints
`MPI: Transfer size (B): ... Bandwidth (GB/s): ...` from 512 KiB through 1 GiB.

Startup picks the query from the device vendor: `MPIX_Query_cuda_support()`
for NVIDIA, `MPIX_Query_hip_support()` or `MPIX_Query_rocm_support()` for
AMD, and `MPIX_Query_ze_support()` for Intel. Intel MPI is decided by its
own `I_MPI_OFFLOAD` (unset or 0 means no). If the library reports no support
for that buffer type, the program aborts before any device pointer is passed
to MPI. If the library reports support, or the headers have no query, one
512 KiB device transfer is checked. A watchdog aborts that transfer if it is
still running after 30 s.

## Non-GPU-aware MPI test (must fail quickly)

Rebuild against host-only MPI, or disable GPU support on Cray:

```bash
export MPICH_GPU_SUPPORT_ENABLED=0
srun -n 2 ./main-mpi
```

```bash
make clean
make MPI_ROOT=/usr/lib/x86_64-linux-gnu/openmpi
/usr/bin/mpirun -n 2 ./main-mpi
```

```bash
# Intel MPI: GPU buffers off (this is the default)
export I_MPI_OFFLOAD=0
/opt/intel/oneapi/2024.1/bin/mpirun -n 2 ./main-mpi
```

**Pass for this test:** the process exits instead of blocking in `MPI_Recv`.
Typical stderr lines:

- `MPI library reports no ... GPU-buffer support` — the library query returned no, or Intel MPI has `I_MPI_OFFLOAD` unset or 0
- `GPU-aware MPI probe failed` — the query was missing or said yes, and the device buffer was wrong
- `GPU-aware MPI probe timed out` — MPI hung on a device pointer; a watchdog aborts within about 30 s
