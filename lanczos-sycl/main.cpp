#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <getopt.h>

#include "graph_io.h"
#include "matrix.h"
#include "linear_algebra.h"
#include "lanczos.h"
#include "cycle_timer.h"
#include "eigen.h"
#include "utils.h"


static string graph_file;
static int node_count = 0;
static int eigen_count = 0;
static bool double_precision = false;

static void usage(const char *program) {
  std::cout << "usage: " << program << " [options]" << std::endl;
  std::cout << "options:" << std::endl;
  std::cout << "  -g --graph <file>" << std::endl;
  std::cout << "  -n --nodes <n>" << std::endl;
  std::cout << "  -k --eigens <k>" << std::endl;
  std::cout << "  -d --double" << std::endl;
}

static void parse_option(int argc, char *argv[]) {
  int opt;
  static struct option long_options[] = {
    { "help", 0, 0, 'h' },
    { "graph", 1, 0, 'g' },
    { "nodes", 1, 0, 'n' },
    { "eigens", 1, 0, 'k' },
    { "double", 0, 0, 'd' },
    { 0, 0, 0, 0 },
  };
  while ((opt = getopt_long(argc, argv, "g:n:k:dh?", long_options, NULL)) != EOF) {
    switch (opt) {
      case 'g':
        graph_file = optarg;
        break;
      case 'n':
        node_count = atoi(optarg);
        break;
      case 'k':
        eigen_count = atoi(optarg);
        break;
      case 'd':
        double_precision = true;
        break;
      case 'h':
      case '?':
      default:
        usage(argv[0]);
        exit(opt == 'h' ? 0 : 1);
    }
  }
  if (graph_file.empty()) {
    std::cerr << argv[0] << ": missing graph file" << std::endl;
    exit(1);
  }
  if (node_count <= 0) {
    std::cerr << argv[0] << ": invalid node count" << std::endl;
    exit(1);
  }
  if (eigen_count <= 0 || eigen_count > node_count) {
    std::cerr << argv[0] << ": invalid eigenvalue count" << std::endl;
    exit(1);
  }
}

template <typename T>
static void run() {
  double start_time = cycle_timer::current_seconds();
  coo_matrix<T> graph = adjacency_matrix_from_graph<T>(node_count, graph_file);
  csr_matrix<T> matrix(graph);
  double end_time = cycle_timer::current_seconds();
  std::cout << "graph load time: " << end_time - start_time << " sec" << std::endl;

  int a = 2;
  int b1 = 64, b2 = 8;
  int skip = 16;
  int k = eigen_count;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::cout << "*** running GPU Lanczos ***" << std::endl;
  for (int steps = a * k + 1; steps < b1 * k; steps += skip) {
    print_vector(gpu_lanczos_eigen(q, matrix, k, steps));
    std::cout << std::endl;
  }
  std::cout << "*** running CPU Lanczos ***" << std::endl;
  for (int steps = a * k + 1; steps < b2 * k; steps += skip) {
    print_vector(lanczos_eigen(matrix, k, steps));
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  parse_option(argc, argv);

  std::cout << std::setprecision(15);
  if (double_precision) {
    run<double>();
  }
  else {
    run<float>();
  }
  return 0;
}
