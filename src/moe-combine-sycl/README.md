# moe-combine-sycl

SYCL build of the multi-GPU MoE combine. Routing, the host reference, and the
reported byte counts match `moe-combine-cuda`. See that README for the workload.

SYCL moves expert outputs with GPU-aware MPI rather than CUDA/HIP IPC. When
more than one rank is used, ranks 0 and 1 run the pingpong benchmark's
device-buffer check before any timed transfer. On Cray MPICH, link the `libmpi_gtl_cuda` or `libmpi_gtl_hsa` that matches
the installed ROCm or CUDA, and export `MPICH_GPU_SUPPORT_ENABLED=1`.
