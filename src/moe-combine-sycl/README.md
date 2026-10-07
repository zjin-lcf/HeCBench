# moe-combine-sycl

SYCL build of the multi-GPU MoE combine. Routing, the host reference, and the
reported byte counts match `moe-combine-cuda`. `combine.hpp` uses the same
entry points as `combine.cuh`: `combine_kernel`, `fill_stage`,
`cross_device_barrier`, `load_raw8`, `decode8`, and `store8`. See that README
for the workload and the bandwidth definitions.

Each rank maps its expert-output buffer with
`sycl::ext::oneapi::experimental::ipc::memory` and reads it from the combine
kernel. MPI carries the IPC handle bytes and the host barriers. It does not
move device buffers.
