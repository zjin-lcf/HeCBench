#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <cxxopts.hpp>
#include <fmt/core.h>
#include "io.hpp"
#include "util.hpp"
#include "host_timer.hpp"

void populateBinaryMatrix(float* input, Dataset* dataset, unsigned int start, unsigned int end);
void populateBinaryTransposeMatrix(float* input, Dataset* dataset, unsigned int partition, unsigned int start, unsigned int end);

int main(int argc, char **argv) {
  try {

    // MKL blas gemm on host or device
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    sycl::queue host(sycl::cpu_selector{}); // exclusive scan on host
    auto policy = oneapi::dpl::execution::make_device_policy(host);

    fmt::print(
        "┌{0:─^{1}}┐\n"
        "│{2: ^{1}}|\n"
        "└{0:─^{1}}┘\n", "", 51, "GPU Matrix multiplication set intersection"
        );

    int multiprocessorCount =
       q.get_device().get_info<sycl::info::device::max_compute_units>();

    int maxThreadsPerBlock =
       q.get_device().get_info<sycl::info::device::max_work_group_size>();

    size_t totalDeviceMemory = 
       q.get_device().get_info<sycl::info::device::global_mem_size>();

    size_t freeDeviceMemory = 0.75f * totalDeviceMemory;

    // arguments
    std::string input;
    std::string output;
    unsigned int partition = 10000;

    cxxopts::Options options(argv[0], "Help");

    options.add_options()
      ("input", "Input dataset path", cxxopts::value<std::string>(input))
      ("output", "Output result path", cxxopts::value<std::string>(output))
      ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
      ("help", "Print help");

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
      fmt::print("{}\n", options.help());
      return 0;
    }

    if (!result.count("input")) {
      fmt::print("{}\n", "No input dataset given! Exiting...");
      return 1;
    }

    Dataset* d = readDataset(input);

    fmt::print(
        "┌{0:─^{1}}┐\n"
        "│{3: ^{2}}|{4: ^{2}}│\n"
        "│{5: ^{2}}|{6: ^{2}}│\n"
        "│{7: ^{2}}|{8: ^{2}}│\n"
        "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
        "Cardinality", d->cardinality,
        "Universe", d->universe,
        "Total elements", d->totalElements, ""
        );

    d->universe++;
    partition = std::min(d->cardinality, partition);

    d->offsets = new unsigned int[d->cardinality];

    // calculate offsets
    oneapi::dpl::exclusive_scan(policy, d->sizes, d->sizes + d->cardinality, d->offsets, 0);

    std::vector<tile> tiles = splitToTiles(d->cardinality, partition);
    std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

    size_t combinations = partition * partition;

    unsigned long long outputMemory = runs.size() * combinations * sizeof(float);

    unsigned long long deviceMemory = (sizeof(float) * d->universe * partition * 2)
      + (sizeof(float) * combinations);

    fmt::print(
        "┌{0:─^{1}}┐\n"
        "│{3: ^{2}}|{4: ^{2}}│\n"
        "│{5: ^{2}}|{6: ^{2}}│\n"
        "│{7: ^{2}}|{8: ^{2}}│\n"
        "│{9: ^{2}}|{10: ^{2}}│\n"
        "└{11:─^{1}}┘\n", "Launch info", 51, 25,
        "Partition", partition,
        "GPU invocations", runs.size(),
        "Required memory (Output)", formatBytes(outputMemory),
        "Required memory (GPU)", formatBytes(deviceMemory),
        ""
        );

    if (deviceMemory > freeDeviceMemory) {
      fmt::print("Error not enough GPU memory ({})!\nExiting...", formatBytes(freeDeviceMemory));
      return 1;
    }

    std::vector<float> counts(runs.size() * combinations);
    float* hostInput = new float[d->universe * partition];
    float* hostInvInput = new float[d->universe * partition];

    memset(hostInput, 0, d->universe * partition * sizeof(float));
    memset(hostInvInput, 0, d->universe * partition * sizeof(float));

    HostTimer hostTimer;
    Interval* device_offload = hostTimer.add("Device offload");

    float* devInput = sycl::malloc_device<float>(d->universe * partition, q);
    float* devInvInput = sycl::malloc_device<float>(d->universe * partition, q);
    float* devOutput = sycl::malloc_device<float>(combinations, q);

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;
    float alpha = 1.f;
    float beta = 0.f;

    unsigned int iter = 0;
    for (unsigned int i = 0; i < tiles.size(); ++i) {
      tile& A = tiles[i];

      populateBinaryMatrix(hostInput, d, A.start, A.end);

      auto copyA = q.memcpy(devInput, hostInput, d->universe * partition * sizeof(float));

      for (unsigned int j = i; j < tiles.size(); ++j) {
        tile& B = tiles[j];
        populateBinaryTransposeMatrix(hostInvInput, d, partition, B.start, B.end);

        auto copyB = q.memcpy(devInvInput, hostInvInput, d->universe * partition * sizeof(float));
          
        sycl::event gemm_done;
        try {
          gemm_done = oneapi::mkl::blas::gemm(q, transA, transB, 
              B.length, A.length, d->universe, // m, n, k, 
              alpha, devInvInput, B.length,    // alpha, A, ldA, 
              devInput, d->universe, beta,     // B, ldB, beta,
              devOutput, B.length,
              {copyA, copyB});
        }
        catch(cl::sycl::exception const& e) {
          std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                    << e.what() << std::endl;
        }

        // transfer result back to host
        q.memcpy(&counts[0] + (iter * combinations), devOutput, sizeof(float) * combinations, gemm_done).wait();

        // clear binary matrix to ensure correctness
        memset(hostInvInput, 0, d->universe * partition * sizeof(float));

        iter++;
      }
      // clear binary matrix to ensure correctness
      memset(hostInput, 0, d->universe * partition * sizeof(float));
    }

    sycl::free(devInput, q);
    sycl::free(devInvInput, q);
    sycl::free(devOutput, q);

    q.wait();

    HostTimer::finish(device_offload);
    hostTimer.print();

    if (!output.empty()) {
      fmt::print("Writing result to file {}\n", output);
      writeResult<float, true>(runs, partition, counts, output);
      fmt::print("Finished\n");
    }

    delete [] hostInput; 
    delete [] hostInvInput; 
    delete [] d->offsets;

  } catch (const cxxopts::OptionException& e) {
    fmt::print("{}\n", e.what());
    return 1;
  }
  return 0;
}


void populateBinaryMatrix(float* input, Dataset* d, unsigned int start, unsigned int end)
{
  unsigned int idx = 0;
  for (unsigned int i = start; i < end; ++i) {
    for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
      input[idx * d->universe + d->elements[j]] = 1.0f;
    }
    idx++;
  }
}

void populateBinaryTransposeMatrix(float* input, Dataset* d, unsigned int partition, unsigned int start, unsigned int end)
{
  unsigned int idx = 0;
  for (unsigned int i = start; i < end; ++i) {
    for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
      input[d->elements[j] * partition + idx] = 1.0f;
    }
    idx++;
  }
}
