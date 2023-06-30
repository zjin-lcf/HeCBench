#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>

const int nContractions = 18;  // the device kernel contains 18 cases

template <typename T>
class tensor_aggregate;

template <typename T>
void contraction (
  sycl::nd_item<1> &item,
  const T *__restrict tensor,
  const T *__restrict adj,
        T *__restrict value,
  const int output_size, 
  const int N, 
  const int nChanels)
{
  int tid = item.get_global_id(0);

  if (tid < output_size) {  
    int C = nChanels;
    int B = N * C;
    int A = N * B;
    int Y = nChanels * nContractions;

    int f = (tid % Y) % nChanels;
    int Case = (tid % Y) / nChanels + 1;
    int y = (tid / Y) % N;
    int x = (tid / Y) / N;

    int a, b, c, d, e;
    T adj_value;

    T sum = (T)0;

    // +-----------+
    // | 1 + 1 + 1 |
    // +-----------+

    // Case 1 (1/50): Fix a, b. Contract c, d, e.
    if (Case == 1) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // Case 2 (3/50): Fix a, d. Contract b, c, e.
    if (Case == 2) {    
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    // Case 3 (5/50): Fix b, c. Contract a, d, e.
    if (Case == 3) {    
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            for (a = 0; a < N; ++a) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }  
    }

    // Case 4 (6/50): Fix b, d. Contract a, c, e.
    if (Case == 4) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // Case 5 (10/50): Fix d, e. Contract a, b, c.
    if (Case == 5) {    
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (a = 0; a < N; ++a) {
          for (b = 0; b < N; ++b) {
            for (c = 0; c < N; ++c) {
              sum += tensor[a * A + b * B + c * C + f] * adj_value;
            }
          }
        }
      }
    }

    // +-------+
    // | 1 + 2 |
    // +-------+

    // Case 6 (11/50): (a, b). Contract (c, d). Singleton (e).
    if (Case == 6) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          c = d;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    // Case 7 (13/50): (a, b). Contract (d, e). Singleton (c).
    if (Case == 7) {
      a = x;
      b = y;

      for (d = 0; d < N; ++d) {
        e = d;
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 8 (17/50): (a, d). Contract (b, c). Singleton (e).
    if (Case == 8) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (b = 0; b < N; ++b) {
            c = b;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 9 (18/50): (a, d). Contract (b, e). Singleton (c).
    if (Case == 9) {
      a = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          for (c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 10 (23/50): (b, c). Contract (a, d). Singleton (e).
    if (Case == 10) {
      b = x;
      c = y;

      for (d = 0; d < N; ++d) {
        for (e = 0; e < N; ++e) {
          adj_value = adj[d * N + e];
          if (adj_value > 0) {
            a = d;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 11 (26/50): (b, d). Contract (a, c). Singleton (e).
    if (Case == 11) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          for (a = 0; a < N; ++a) {
            c = a;
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 12 (27/50): (b, d). Contract (a, e). Singleton (c).
    if (Case == 12) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 13 (28/50): (b, d). Contract (c, e). Singleton (a).
    if (Case == 13) {
      b = x;
      d = y;

      for (e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          c = e;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 14 (38/50): (d, e). Contract (a, b). Singleton (c).
    if (Case == 14) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          for (int c = 0; c < N; ++c) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // Case 15 (40/50): (d, e). Contract (b, c). Singleton (a).
    if (Case == 15) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int b = 0; b < N; ++b) {
          c = b;
          for (int a = 0; a < N; ++a) {
            sum += tensor[a * A + b * B + c * C + f] * adj_value;
          }
        }
      }
    }

    // +---+
    // | 3 |
    // +---+

    // Case 16 (43/50): (a, d). Contract (b, c, e).
    if (Case == 16) {
      a = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          b = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }  

    // Case 17 (46/50): (b, d). Contract (a, c, e).
    if (Case == 17) {
      b = x;
      d = y;

      for (int e = 0; e < N; ++e) {
        adj_value = adj[d * N + e];
        if (adj_value > 0) {
          a = e;
          c = e;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    // Case 18 (50/50): (d, e). Contract (a, b, c).
    if (Case == 18) {
      d = x;
      e = y;

      adj_value = adj[d * N + e];
      if (adj_value > 0) {
        for (int a = 0; a < N; ++a) {
          b = a;
          c = a;
          sum += tensor[a * A + b * B + c * C + f] * adj_value;
        }
      }
    }

    value[tid] = sum;
  }
}

int rounded_division(int number1, int number2) {
  if (number1 % number2 == 0)
    return number1 / number2;
  return number1 / number2 + 1;
}

template <typename T>
void contract (sycl::queue &q, const int max_N, const int max_C, const int repeat) {
  // tensor
  const size_t tensor_size = max_N * max_N * max_N * max_C * sizeof(T);
  const size_t tensor_size_byte = tensor_size * sizeof(T);

  T* tensor_value = (T*) malloc (tensor_size_byte);
  for (size_t i = 0; i < max_N * max_N * max_N * max_C; i++)
    tensor_value[i] = 1;

  T* d_tensor_value = (T*) sycl::malloc_device (tensor_size_byte, q);
  q.memcpy(d_tensor_value, tensor_value, tensor_size_byte);

  // adjacency matrix
  const size_t adj_size = max_N * max_N;
  const size_t adj_size_byte = adj_size * sizeof(T);
  
  // longest kernel time occurs when all values in adj_value are positive
  T* adj_value = (T*) malloc (adj_size_byte);
  for (int i = 0; i < adj_size; i++) adj_value[i] = 1;

  T* d_adj_value = (T*) sycl::malloc_device (adj_size_byte, q);
  q.memcpy(d_adj_value, adj_value, adj_size_byte);

  // output value 
  const size_t output_size = max_N * max_N * max_C * nContractions;
  const size_t output_size_byte = max_N * max_N * max_C * nContractions * sizeof(T);

  T* value = (T*) malloc (output_size_byte);
  T* d_value = (T*) sycl::malloc_device (output_size_byte, q);

  // launch kernel
  const int nThreads = 256;
  sycl::range<1> gws (rounded_division(output_size, nThreads) * nThreads);
  sycl::range<1> lws (nThreads);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class tensor_aggregate<T>>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        contraction<T>(item, d_tensor_value, d_adj_value, d_value,
                       output_size, max_N, max_C);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(value, d_value, output_size_byte).wait();

  double checksum = 0;
  for (size_t i = 0; i < output_size; i++) checksum += value[i];
  printf("Checksum: %lf min:%lf max:%lf\n", checksum, 
         *std::min_element(value, value+output_size),
         *std::max_element(value, value+output_size));

  free(value);
  free(tensor_value);
  free(adj_value);
  sycl::free(d_value, q);
  sycl::free(d_tensor_value, q);
  sycl::free(d_adj_value, q);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dimension> <repeat>\n", argv[0]);
    return 1;
  }
 
  int max_N = atoi(argv[1]);
  int max_C = nContractions;
  int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  contract<float>(q, max_N, max_C, repeat);
  contract<double>(q, max_N, max_C, repeat);

  return 0;
}
