#include "collectives.h"
#include "timer.h"

#include <mpi.h>
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <iostream>
#include <vector>

void TestCollectivesGPU(std::vector<size_t>& sizes, std::vector<size_t>& iterations) {
    // Get the local rank, which gets us the GPU we should be using.
    //
    // We must do this before initializing MPI, because initializing MPI requires having the right
    // GPU context, so we use environment variables from our MPI implementation to determine the
    // local rank.
    // 
    // OpenMPI usually provides OMPI_COMM_WORLD_LOCAL_RANK, which we read. If you use SLURM with
    // OpenMPI, then SLURM instead provides SLURM_LOCALID. In this case, make sure to use `srun` or
    // `sbatch` and not `mpirun` to run your application.
    //
    // Remember that in order for this to work, you must have a GPU-enabled CUDA-aware MPI build.
    // Otherwise, this will result in a segfault, when MPI tries to read from a GPU memory pointer.
    char* env_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(env_str == NULL) {
        env_str = std::getenv("SLURM_LOCALID");
    }
    if(env_str == NULL) {
        throw std::runtime_error("Could not find OMPI_COMM_WORLD_LOCAL_RANK or SLURM_LOCALID!");
    }

    // Assume that the environment variable has an integer in it.
    int mpi_local_rank = std::stoi(std::string(env_str));
    InitCollectives(mpi_local_rank);

    // Get the MPI size and rank.
    int mpi_size;
    if(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_size failed with an error");

    int mpi_rank;
    if(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_rank failed with an error");

    hipError_t err;

    timer::Timer timer;
    for(size_t i = 0; i < sizes.size(); i++) {
        auto size = sizes[i];
        auto iters = iterations[i];

        float* cpu_data = new float[size];

        float* data;
        err = hipMalloc(&data, sizeof(float) * size);
        if(err != hipSuccess) { throw std::runtime_error("hipMalloc failed with an error"); }

        float seconds = 0.0f;
        for(size_t iter = 0; iter < iters; iter++) {
            // Initialize data as a block of ones, which makes it easy to check for correctness.
            for(size_t j = 0; j < size; j++) {
                cpu_data[j] = 1.0f;
            }

            err = hipMemcpy(data, cpu_data, sizeof(float) * size, hipMemcpyHostToDevice);
            if(err != hipSuccess) { throw std::runtime_error("hipMemcpy failed with an error"); }

            float* output;
            timer.start();
            RingAllreduce(data, size, &output);
            seconds += timer.seconds();

            err = hipMemcpy(cpu_data, output, sizeof(float) * size, hipMemcpyDeviceToHost);
            if(err != hipSuccess) { throw std::runtime_error("hipMemcpy failed with an error"); }

            // Check that we get the expected result.
            for(size_t j = 0; j < size; j++) {
                if(cpu_data[j] != (float) mpi_size) {
                    std::cerr << "Unexpected result from allreduce: " << cpu_data[j] << std::endl;
                    return;
                }
            }
            err = hipFree(output);
            if(err != hipSuccess) { throw std::runtime_error("hipFree failed with an error"); }
        }
        if(mpi_rank == 0) {
            std::cout << "Verified allreduce for size "
                << size
                << " ("
                << seconds * 1e6 / iters
                << " us per iteration)" << std::endl;
        }

        err = hipFree(data);
        if(err != hipSuccess) { throw std::runtime_error("hipFree failed with an error"); }
        delete[] cpu_data;
    }
}

// Test program for baidu-allreduce collectives, should be run using `mpirun`.
int main(int argc, char** argv) {
    // Buffer sizes used for tests.
    std::vector<size_t> buffer_sizes = {
        0, 32, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 8388608, 67108864, 536870912
    };

    // Number of iterations to run for each buffer size.
    std::vector<size_t> iterations = {
        100000, 100000, 100000, 100000,
        1000, 1000, 1000, 1000,
        100, 50, 10, 1
    };

    TestCollectivesGPU(buffer_sizes, iterations);

    // Finalize to avoid any MPI errors on shutdown.
    MPI_Finalize();

    return 0;
}
