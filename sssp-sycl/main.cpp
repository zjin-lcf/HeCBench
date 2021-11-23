/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba,
 *        University of Illinois nor the names of its contributors may be used
 *        to endorse or promote products derived from this Software without
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <unistd.h>
#include <thread>
#include <assert.h>
#include "common.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"


// Params ---------------------------------------------------------------------
struct Params {

  int         device;
  int         n_gpu_threads;
  int         n_gpu_blocks;
  int         n_threads;
  int         n_warmup;
  int         n_reps;
  const char *file_name;
  const char *comparison_file;
  int         switching_limit;

  Params(int argc, char **argv) {
    device          = 0;
    n_gpu_threads    = 256;
    n_gpu_blocks   = 8;
    n_threads       = 2;
    n_warmup        = 1;
    n_reps          = 1;
    file_name       = "input/NYR_input.dat";
    comparison_file = "output/NYR_bfs.out";
    switching_limit = 128;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:t:w:r:f:c:l:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'd': device          = atoi(optarg); break;
        case 'i': n_gpu_threads    = atoi(optarg); break;
        case 'g': n_gpu_blocks   = atoi(optarg); break;
        case 't': n_threads       = atoi(optarg); break;
        case 'w': n_warmup        = atoi(optarg); break;
        case 'r': n_reps          = atoi(optarg); break;
        case 'f': file_name       = optarg; break;
        case 'c': comparison_file = optarg; break;
        case 'l': switching_limit = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
    assert(n_gpu_threads > 0 && "Invalid # of device threads!");
    assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
    assert(n_threads > 0 && "Invalid # of host threads!");
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./sssp [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -d <D>    CUDA device ID (default=0)"
        "\n    -i <I>    # of device threads per block (default=256)"
        "\n    -g <G>    # of device blocks (default=8)"
        "\n    -t <T>    # of host threads (default=2)"
        "\n    -w <W>    # of untimed warmup iterations (default=1)"
        "\n    -r <R>    # of timed repetition iterations (default=1)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    name of input file with control points (default=input/NYR_input.dat)"
        "\n    -c <C>    comparison file (default=output/NYR_bfs_BFS.out)"
        "\n    -l <L>    switching limit (default=128)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input_size(int &n_nodes, int &n_edges, const Params &p) {
  FILE *fp = fopen(p.file_name, "r");
  fscanf(fp, "%d", &n_nodes);
  fscanf(fp, "%d", &n_edges);
  if(fp)
    fclose(fp);
}

void read_input(int &source, Node *&h_nodes, Edge *&h_edges, const Params &p) {

  int   start, edgeno;
  int   n_nodes, n_edges;
  int   id, cost;
  FILE *fp = fopen(p.file_name, "r");

  fscanf(fp, "%d", &n_nodes);
  fscanf(fp, "%d", &n_edges);
  fscanf(fp, "%d", &source);
  printf("Number of nodes = %d\t", n_nodes);
  printf("Number of edges = %d\t", n_edges);

  // initalize the memory: Nodes
  for(int i = 0; i < n_nodes; i++) {
    fscanf(fp, "%d %d", &start, &edgeno);
    h_nodes[i].x = start;
    h_nodes[i].y = edgeno;
  }
#if PRINT_ALL
  for(int i = 0; i < n_nodes; i++) {
    printf("%d, %d\n", h_nodes[i].x, h_nodes[i].y);
  }
#endif

  // initalize the memory: Edges
  for(int i = 0; i < n_edges; i++) {
    fscanf(fp, "%d", &id);
    fscanf(fp, "%d", &cost);
    h_edges[i].x = id;
    h_edges[i].y = -cost;
  }
  if(fp)
    fclose(fp);
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

  const Params p(argc, argv);
  //CUDASetup    setcuda(p.device);
  Timer        timer;
  //cudaError_t  cudaStatus;
  
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  // Allocate
  int n_nodes, n_edges;
  read_input_size(n_nodes, n_edges, p);
  timer.start("Allocation");
  Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
  buffer<Node, 1> d_nodes(n_nodes);
  Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
  buffer<Edge, 1> d_edges (n_edges);
  std::atomic_int *h_color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
  buffer<int, 1> d_color (n_nodes);
  std::atomic_int *h_cost  = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
  buffer<int, 1> d_cost (n_nodes);
  int *            h_q1    = (int *)malloc(n_nodes * sizeof(int));
  buffer<int, 1> d_q1 (n_nodes);
  int *            h_q2    = (int *)malloc(n_nodes * sizeof(int));
  buffer<int, 1> d_q2 (n_nodes);
  std::atomic_int  h_head[1];
  buffer<int, 1> d_head (1);
  std::atomic_int  h_tail[1];
  buffer<int, 1> d_tail (1);
  std::atomic_int  h_threads_end[1];
  buffer<int, 1> d_threads_end (1);
  std::atomic_int  h_threads_run[1];
  buffer<int, 1> d_threads_run (1);
  int              h_num_t[1];
  buffer<int, 1> d_num_t (1);
  int              h_overflow[1];
  buffer<int, 1> d_overflow (1);
  std::atomic_int  h_gray_shade[1];
  buffer<int, 1> d_gray_shade (1);
  std::atomic_int  h_iter[1];
  buffer<int, 1> d_iter (1);
  q.wait();
  timer.stop("Allocation");

  // Initialize
  timer.start("Initialization");
  const int max_gpu_threads = 256; 
  int source;
  read_input(source, h_nodes, h_edges, p);
  for(int i = 0; i < n_nodes; i++) {
    h_cost[i].store(INF);
  }
  h_cost[source].store(0);
  for(int i = 0; i < n_nodes; i++) {
    h_color[i].store(WHITE);
  }
  h_tail[0].store(0);
  h_head[0].store(0);
  h_threads_end[0].store(0);
  h_threads_run[0].store(0);
  h_q1[0] = source;
  h_iter[0].store(0);
  h_overflow[0] = 0;
  h_gray_shade[0].store(GRAY0);
  timer.stop("Initialization");
  timer.print("Initialization", 1);

  // Copy to device
  timer.start("Copy To Device");
  q.submit([&] (handler &h) {
    auto d_nodes_acc = d_nodes.get_access<sycl_write>(h);
    h.copy(h_nodes, d_nodes_acc);
  });
  q.submit([&] (handler &h) {
    auto d_edges_acc = d_edges.get_access<sycl_write>(h);
    h.copy(h_edges, d_edges_acc);
  });
  q.wait();
  timer.stop("Copy To Device");

  for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

    // Reset
    for(int i = 0; i < n_nodes; i++) {
      h_cost[i].store(INF);
    }
    h_cost[source].store(0);
    for(int i = 0; i < n_nodes; i++) {
      h_color[i].store(WHITE);
    }
    h_tail[0].store(0);
    h_head[0].store(0);
    h_threads_end[0].store(0);
    h_threads_run[0].store(0);
    h_q1[0] = source;
    h_iter[0].store(0);
    h_overflow[0] = 0;
    h_gray_shade[0].store(GRAY0);

    if(rep >= p.n_warmup)
      timer.start("Kernel");

    // Run first iteration in master CPU thread
    h_num_t[0] = 1;
    int pid;
    int index_i, index_o;
    for(index_i = 0; index_i < h_num_t[0]; index_i++) {
      pid = h_q1[index_i];
      h_color[pid].store(BLACK);
      int cur_cost = h_cost[pid].load();
      for(int i = h_nodes[pid].x; i < (h_nodes[pid].y + h_nodes[pid].x); i++) {
        int id   = h_edges[i].x;
        int cost = h_edges[i].y;
        cost += cur_cost;
        h_cost[id].store(cost);
        h_color[id].store(GRAY0);
        index_o       = h_tail[0].fetch_add(1);
        h_q2[index_o] = id;
      }
    }
    h_num_t[0] = h_tail[0].load();
    h_tail[0].store(0);
    h_threads_run[0].fetch_add(1);
    h_gray_shade[0].store(GRAY1);
    h_iter[0].fetch_add(1);
    if(rep >= p.n_warmup)
      timer.stop("Kernel");

    // Pointers to input and output queues
    int * h_qin  = h_q2;
    int * h_qout = h_q1;

    const int CPU_EXEC = (p.n_threads > 0) ? 1 : 0;
    const int GPU_EXEC = (p.n_gpu_blocks > 0 && p.n_gpu_threads > 0) ? 1 : 0;

    // Run subsequent iterations on CPU or GPU until number of input queue elements is 0
    while(*h_num_t != 0) {

      if((*h_num_t < p.switching_limit || GPU_EXEC == 0) &&
          CPU_EXEC == 1) { // If the number of input queue elements is lower than switching_limit

        if(rep >= p.n_warmup)
          timer.start("Kernel");

        // Continue until switching_limit condition is not satisfied
        while((*h_num_t != 0) && (*h_num_t < p.switching_limit || GPU_EXEC == 0) && CPU_EXEC == 1) {

          // Swap queues
          if(h_iter[0] % 2 == 0) {
            h_qin  = h_q1;
            h_qout = h_q2;
          } else {
            h_qin  = h_q2;
            h_qout = h_q1;
          }

          std::thread main_thread(run_cpu_threads, h_nodes, h_edges, h_cost, h_color, h_qin, h_qout, h_num_t,
              h_head, h_tail, h_threads_end, h_threads_run, h_gray_shade, h_iter, p.n_threads,
              p.switching_limit, GPU_EXEC);
          main_thread.join();

          h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
          h_tail[0].store(0);
          h_head[0].store(0);
          if(h_iter[0].load() % 2 == 0)
            h_gray_shade[0].store(GRAY0);
          else
            h_gray_shade[0].store(GRAY1);
        }

        if(rep >= p.n_warmup)
          timer.stop("Kernel");

      } else if((*h_num_t >= p.switching_limit || CPU_EXEC == 0) && GPU_EXEC == 1) { 
        // If the number of input queue elements is higher than or equal to switching_limit

        if(rep >= p.n_warmup)
          timer.start("Copy To Device");

        q.submit([&] (handler &h) {
          auto d_cost_acc = d_cost.get_access<sycl_write>(h);
          h.copy(h_cost, d_cost_acc);
        });
        q.submit([&] (handler &h) {
          auto d_color_acc = d_color.get_access<sycl_write>(h);
          h.copy(h_color, d_color_acc);
        });
        q.submit([&] (handler &h) {
          auto d_threads_run_acc = d_threads_run.get_access<sycl_write>(h);
          h.copy(h_threads_run, d_threads_run_acc);
        });
        q.submit([&] (handler &h) {
          auto d_threads_end_acc = d_threads_end.get_access<sycl_write>(h);
          h.copy(h_threads_end, d_threads_end_acc);
        });
        q.submit([&] (handler &h) {
          auto d_overflow_acc = d_overflow.get_access<sycl_write>(h);
          h.copy(h_overflow, d_overflow_acc);
        });
        q.submit([&] (handler &h) {
          auto d_q1_acc = d_q1.get_access<sycl_write>(h);
          h.copy(h_q1, d_q1_acc);
        });
        q.submit([&] (handler &h) {
          auto d_q2_acc = d_q2.get_access<sycl_write>(h);
          h.copy(h_q2, d_q2_acc);
        });
        q.submit([&] (handler &h) {
          auto d_iter_acc = d_iter.get_access<sycl_write>(h);
          h.copy(h_iter, d_iter_acc);
        });
        q.wait();
        if(rep >= p.n_warmup)
          timer.stop("Copy To Device");


        // Continue until switching_limit condition is not satisfied
        while((*h_num_t != 0) && (*h_num_t >= p.switching_limit || CPU_EXEC == 0) && GPU_EXEC == 1) {

          // Swap queues
	  auto d_qin = (h_iter[0] % 2 == 0) ? &d_q1 : &d_q2;
	  auto d_qout = (h_iter[0] % 2 == 0) ? &d_q2 : &d_q1;

          if(rep >= p.n_warmup)
            timer.start("Copy To Device");

          q.submit([&] (handler &h) {
            auto d_num_acc = d_num_t.get_access<sycl_write>(h);
            h.copy(h_num_t, d_num_acc);
          });
          q.submit([&] (handler &h) {
            auto d_tail_acc = d_tail.get_access<sycl_write>(h);
            h.copy(h_tail, d_tail_acc);
          });
          q.submit([&] (handler &h) {
            auto d_head_acc = d_head.get_access<sycl_write>(h);
            h.copy(h_head, d_head_acc);
          });
          q.submit([&] (handler &h) {
            auto d_gray_shade_acc = d_gray_shade.get_access<sycl_write>(h);
            h.copy(h_gray_shade, d_gray_shade_acc);
          });
          q.wait();
          if(rep >= p.n_warmup)
            timer.stop("Copy To Device");

          if(rep >= p.n_warmup)
            timer.start("Kernel");
          assert(p.n_gpu_threads <= max_gpu_threads && 
              "The thread block size is greater than the maximum thread block size that can be used on this device");
          int blocks = p.n_gpu_blocks;
          int threads = p.n_gpu_threads;
          q.submit([&] (handler &h) {
            auto graph_nodes_av = d_nodes.get_access<sycl_read>(h);
            auto graph_edges_av = d_edges.get_access<sycl_read>(h);
            auto cost = d_cost.get_access<sycl_atomic>(h);
            auto color = d_color.get_access<sycl_atomic>(h);
            auto q1 = (*d_qin).get_access<sycl_read>(h);
            auto q2 = (*d_qout).get_access<sycl_write>(h);
            auto n_t = d_num_t.get_access<sycl_atomic>(h);
            auto head = d_head.get_access<sycl_atomic>(h);
            auto tail = d_tail.get_access<sycl_atomic>(h);
            auto threads_end = d_threads_end.get_access<sycl_read>(h);
            auto threads_run = d_threads_run.get_access<sycl_read>(h);
            auto overflow = d_overflow.get_access<sycl_read_write>(h);
            auto gray_shade = d_gray_shade.get_access<sycl_atomic>(h);
            auto iter = d_iter.get_access<sycl_atomic>(h);
            accessor<int, 1, sycl_atomic, access::target::local> tail_bin (1, h);
            accessor<int, 1, sycl_read_write, access::target::local> l_mem (W_QUEUE_SIZE + 2, h);
            h.parallel_for(nd_range<1>(range<1>(blocks*threads), range<1>(threads)), [=] (nd_item<1> item) {
              int* l_q2 = &l_mem[0];
              int* shift = &l_mem[W_QUEUE_SIZE];
              int* base = &l_mem[W_QUEUE_SIZE+1];

              const int tid     = item.get_local_id(0); 
              const int gtid    = item.get_global_id(0); 
              const int MAXWG   = item.get_global_range(0);
              const int WG_SIZE = item.get_group_range(0);

              int iter_local = atomic_fetch_add(iter[0], 0); //atomicAdd(&iter[0], 0);
              int n_t_local = atomic_fetch_add(n_t[0], 0); //atomicAdd(n_t, 0);
              int gray_shade_local = atomic_fetch_add(gray_shade[0], 0); //atomicAdd(&gray_shade[0], 0);

              if(tid == 0) {
                // Reset queue
                tail_bin[0].store(0);
              }

              // Fetch frontier elements from the queue
              if(tid == 0)
                *base = atomic_fetch_add(head[0], WG_SIZE); //atomicAdd(&head[0], WG_SIZE);
              item.barrier(access::fence_space::local_space);

              int my_base = *base;
              while(my_base < n_t_local) {

                // If local queue might overflow
                if(atomic_load(tail_bin[0]) >= W_QUEUE_SIZE / 2) {
                  if(tid == 0) {
                    // Add local tail_bin to tail
                    *shift = atomic_fetch_add(tail[0], atomic_load(tail_bin[0])); //atomicAdd(&tail[0], *tail_bin);
                  }
                  item.barrier(access::fence_space::local_space);
                  int local_shift = tid;
                  while(local_shift < atomic_load(tail_bin[0])) {
                    q2[*shift + local_shift] = l_q2[local_shift];
                    // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                    local_shift += WG_SIZE;
                  }
                  item.barrier(access::fence_space::local_space);
                  if(tid == 0) {
                    // Reset local queue
                    tail_bin[0].store(0);
                  }
                  item.barrier(access::fence_space::local_space);
                }

                if(my_base + tid < n_t_local && overflow[0] == 0) {
                  // Visit a node from the current frontier
                  int pid = q1[my_base + tid];
                  //////////////// Visit node ///////////////////////////
                  atomic_exchange(color[pid], BLACK); //atomicExch(&color[pid], BLACK); // Node visited
                  int  cur_cost = atomic_fetch_add(cost[pid], 0); //atomicAdd(&cost[pid], 0); // Look up shortest-path distance to this node
                  Node cur_node;
                  cur_node.x = graph_nodes_av[pid].x;
                  cur_node.y = graph_nodes_av[pid].y;
                  Edge cur_edge;
                  // For each outgoing edge
                  for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
                    cur_edge.x = graph_edges_av[i].x;
                    cur_edge.y = graph_edges_av[i].y;
                    int id     = cur_edge.x;
                    int cost_local   = cur_edge.y;
                    cost_local += cur_cost;
                    int orig_cost = atomic_fetch_max(cost[id], cost_local); //atomicMax(&cost[id], cost_local);
                    if(orig_cost < cost_local) {
                      int old_color = atomic_fetch_max(color[id], gray_shade_local); //atomicMax(&color[id], gray_shade_local);
                      if(old_color != gray_shade_local) {
                        // Push to the queue
                        int tail_index = atomic_fetch_add(tail_bin[0], 1); // atomicAdd(tail_bin, 1);
                        if(tail_index >= W_QUEUE_SIZE) {
                          overflow[0] = 1;
                        } else
                          l_q2[tail_index] = id;
                      }
                    }
                  }
                }

                if(tid == 0)
                  *base = atomic_fetch_add(head[0], WG_SIZE); //atomicAdd(&head[0], WG_SIZE); // Fetch more frontier elements from the queue
                item.barrier(access::fence_space::local_space);
                my_base = *base;
              }
              /////////////////////////////////////////////////////////
              // Compute size of the output and allocate space in the global queue
              if(tid == 0) {
                *shift = atomic_fetch_add(tail[0], atomic_load(tail_bin[0])); //atomicAdd(&tail[0], *tail_bin);
              }
              item.barrier(access::fence_space::local_space);
              ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
              int local_shift = tid;
              while(local_shift < atomic_load(tail_bin[0])) {
                q2[*shift + local_shift] = l_q2[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
              }
              //////////////////////////////////////////////////////////////////////////

              if(gtid == 0) {
                atomic_fetch_add(iter[0], 1); //atomicAdd(&iter[0], 1);
              }
            });
          });

          q.wait();
          if(rep >= p.n_warmup)
            timer.stop("Kernel");

          if(rep >= p.n_warmup)
            timer.start("Copy Back and Merge");
          q.submit([&] (handler &h) {
            auto d_tail_acc = d_tail.get_access<sycl_read>(h);
            h.copy(d_tail_acc, h_tail);
          });
          q.submit([&] (handler &h) {
            auto d_iter_acc = d_iter.get_access<sycl_read>(h);
            h.copy(d_iter_acc, h_iter);
          });
	  q.wait();

          if(rep >= p.n_warmup)
            timer.stop("Copy Back and Merge");

          h_num_t[0] = h_tail[0].load(); // Number of elements in output queue
          h_tail[0].store(0);
          h_head[0].store(0);
          if(h_iter[0].load() % 2 == 0)
            h_gray_shade[0].store(GRAY0);
          else
            h_gray_shade[0].store(GRAY1);
        }

        if(rep >= p.n_warmup)
          timer.start("Copy Back and Merge");
        q.submit([&] (handler &h) {
          auto d_cost_acc = d_cost.get_access<sycl_read>(h);
          h.copy(d_cost_acc, h_cost);
        });
        q.submit([&] (handler &h) {
          auto d_color_acc = d_color.get_access<sycl_read>(h);
          h.copy(d_color_acc, h_color);
        });
        q.submit([&] (handler &h) {
          auto d_threads_run_acc = d_threads_run.get_access<sycl_read>(h);
          h.copy(d_threads_run_acc, h_threads_run);
        });
        q.submit([&] (handler &h) {
          auto d_threads_end_acc = d_threads_end.get_access<sycl_read>(h);
          h.copy(d_threads_end_acc, h_threads_end);
        });
        q.submit([&] (handler &h) {
          auto d_overflow_acc = d_overflow.get_access<sycl_read>(h);
          h.copy(d_overflow_acc, h_overflow);
        });
        q.submit([&] (handler &h) {
          auto d_q1_acc = d_q1.get_access<sycl_read>(h);
          h.copy(d_q1_acc, h_q1);
        });
        q.submit([&] (handler &h) {
          auto d_q2_acc = d_q2.get_access<sycl_read>(h);
          h.copy(d_q2_acc, h_q2);
        });
        q.wait();
        if(rep >= p.n_warmup)
          timer.stop("Copy Back and Merge");
      }
    }

  } // end of iteration
  timer.print("Allocation", 1);
  timer.print("Copy To Device", p.n_reps);
  timer.print("Kernel", p.n_reps);
  timer.print("Copy Back and Merge", p.n_reps);

  // Verify answer
  bool ok = verify(h_cost, n_nodes, p.comparison_file);

  // Free memory
  timer.start("Deallocation");
  free(h_nodes);
  free(h_edges);
  free(h_color);
  free(h_cost);
  free(h_q1);
  free(h_q2);
  timer.stop("Deallocation");
  timer.print("Deallocation", 1);

  if (ok) printf("Test Passed\n");
  return 0;
}
