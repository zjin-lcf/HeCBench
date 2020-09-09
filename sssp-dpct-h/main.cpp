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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <unistd.h>
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

// CUDA kernel ------------------------------------------------------------------------------------------
void SSSP_gpu(Node *graph_nodes_av, Edge *graph_edges_av, int *cost,
    int *color, int *q1, int *q2, int *n_t,
    int *head, int *tail, int *threads_end, int *threads_run,
    int *overflow, int *gray_shade, int *iter, int LIMIT, const int CPU,
    sycl::nd_item<3> item_ct1, int *l_mem, int *tail_bin) {

    int* l_q2 = (int*)&l_mem[0];
    int* shift = (int*)&l_mem[W_QUEUE_SIZE];
    int* base = (int*)&l_mem[W_QUEUE_SIZE+1];

    const int tid = item_ct1.get_local_id(2);
    const int gtid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
    const int MAXWG = item_ct1.get_group_range(2);
    const int WG_SIZE = item_ct1.get_local_range().get(2);

    int iter_local =
        sycl::atomic<int>(sycl::global_ptr<int>(&iter[0])).fetch_add(0);

    int n_t_local = sycl::atomic<int>(sycl::global_ptr<int>(n_t)).fetch_add(0);

    int gray_shade_local =
        sycl::atomic<int>(sycl::global_ptr<int>(&gray_shade[0])).fetch_add(0);

    if(tid == 0) {
        // Reset queue
        *tail_bin = 0;
    }

    // Fetch frontier elements from the queue
    if(tid == 0)
        *base = sycl::atomic<int>(sycl::global_ptr<int>(&head[0]))
                    .fetch_add(WG_SIZE);
    item_ct1.barrier();

    int my_base = *base;
    while(my_base < n_t_local) {

        // If local queue might overflow
        if (*tail_bin >= W_QUEUE_SIZE / 2) {
            if(tid == 0) {
                // Add local tail_bin to tail
                *shift = sycl::atomic<int>(sycl::global_ptr<int>(&tail[0]))
                             .fetch_add(*tail_bin);
            }
            item_ct1.barrier();
            int local_shift = tid;
            while (local_shift < *tail_bin) {
                q2[*shift + local_shift] = l_q2[local_shift];
                // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
                local_shift += WG_SIZE;
            }
            item_ct1.barrier();
            if(tid == 0) {
                // Reset local queue
                *tail_bin = 0;
            }
            item_ct1.barrier();
        }

        if(my_base + tid < n_t_local && *overflow == 0) {
            // Visit a node from the current frontier
            int pid = q1[my_base + tid];
            //////////////// Visit node ///////////////////////////
            dpct::atomic_exchange(&color[pid], BLACK); // Node visited
            int cur_cost =
                sycl::atomic<int>(sycl::global_ptr<int>(&cost[pid]))
                    .fetch_add(
                        0); // Look up shortest-path distance to this node
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
                int orig_cost =
                    sycl::atomic<int>(sycl::global_ptr<int>(&cost[id]))
                        .fetch_max(cost_local);
                if(orig_cost < cost_local) {
                    int old_color =
                        sycl::atomic<int>(sycl::global_ptr<int>(&color[id]))
                            .fetch_max(gray_shade_local);
                    if(old_color != gray_shade_local) {
                        // Push to the queue
                        int tail_index =
                            sycl::atomic<
                                int, sycl::access::address_space::local_space>(
                                sycl::local_ptr<int>(tail_bin))
                                .fetch_add(1);
                        if(tail_index >= W_QUEUE_SIZE) {
                            *overflow = 1;
                        } else
                            l_q2[tail_index] = id;
                    }
                }
            }
        }

        if(tid == 0)
            *base =
                sycl::atomic<int>(sycl::global_ptr<int>(&head[0]))
                    .fetch_add(
                        WG_SIZE); // Fetch more frontier elements from the queue
        item_ct1.barrier();
        my_base = *base;
    }
    /////////////////////////////////////////////////////////
    // Compute size of the output and allocate space in the global queue
    if(tid == 0) {
        *shift = sycl::atomic<int>(sycl::global_ptr<int>(&tail[0]))
                     .fetch_add(*tail_bin);
    }
    item_ct1.barrier();
    ///////////////////// CONCATENATE INTO GLOBAL MEMORY /////////////////////
    int local_shift = tid;
    while (local_shift < *tail_bin) {
        q2[*shift + local_shift] = l_q2[local_shift];
        // Multiple threads are copying elements at the same time, so we shift by multiple elements for next iteration
        local_shift += WG_SIZE;
    }
    //////////////////////////////////////////////////////////////////////////

    if(gtid == 0) {
        sycl::atomic<int>(sycl::global_ptr<int>(&iter[0])).fetch_add(1);
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();

    const Params p(argc, argv);
    Timer        timer;

    // Allocate
    int n_nodes, n_edges;
    read_input_size(n_nodes, n_edges, p);
    timer.start("Allocation");
    Node * h_nodes = (Node *)malloc(sizeof(Node) * n_nodes);
    Node * d_nodes;
    dpct::dpct_malloc((void **)&d_nodes, sizeof(Node) * n_nodes);
    Edge * h_edges = (Edge *)malloc(sizeof(Edge) * n_edges);
    Edge * d_edges;
    dpct::dpct_malloc((void **)&d_edges, sizeof(Edge) * n_edges);
    std::atomic_int *h_color = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int * d_color;
    dpct::dpct_malloc((void **)&d_color, sizeof(int) * n_nodes);
    std::atomic_int *h_cost  = (std::atomic_int *)malloc(sizeof(std::atomic_int) * n_nodes);
    int * d_cost;
    dpct::dpct_malloc((void **)&d_cost, sizeof(int) * n_nodes);
    int *            h_q1    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q1;
    dpct::dpct_malloc((void **)&d_q1, sizeof(int) * n_nodes);
    int *            h_q2    = (int *)malloc(n_nodes * sizeof(int));
    int * d_q2;
    dpct::dpct_malloc((void **)&d_q2, sizeof(int) * n_nodes);
    std::atomic_int  h_head[1];
    int * d_head;
    dpct::dpct_malloc((void **)&d_head, sizeof(int));
    std::atomic_int  h_tail[1];
    int * d_tail;
    dpct::dpct_malloc((void **)&d_tail, sizeof(int));
    std::atomic_int  h_threads_end[1];
    int * d_threads_end;
    dpct::dpct_malloc((void **)&d_threads_end, sizeof(int));
    std::atomic_int  h_threads_run[1];
    int * d_threads_run;
    dpct::dpct_malloc((void **)&d_threads_run, sizeof(int));
    int              h_num_t[1];
    int * d_num_t;
    dpct::dpct_malloc((void **)&d_num_t, sizeof(int));
    int              h_overflow[1];
    int * d_overflow;
    dpct::dpct_malloc((void **)&d_overflow, sizeof(int));
    std::atomic_int  h_gray_shade[1];
    int * d_gray_shade;
    dpct::dpct_malloc((void **)&d_gray_shade, sizeof(int));
    std::atomic_int  h_iter[1];
    int * d_iter;
    dpct::dpct_malloc((void **)&d_iter, sizeof(int));
    dev_ct1.queues_wait_and_throw();
    //CUDA_ERR();
    //ALLOC_ERR(h_nodes, h_edges, h_color, h_cost, h_q1, h_q2);
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
    dpct::dpct_memcpy(d_nodes, h_nodes, sizeof(Node) * n_nodes,
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_edges, h_edges, sizeof(Edge) * n_edges,
                      dpct::host_to_device);
    dev_ct1.queues_wait_and_throw();
    //CUDA_ERR();
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
        int * d_qin; //  = d_q2;
        int * d_qout; // = d_q1;

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

            } else if((*h_num_t >= p.switching_limit || CPU_EXEC == 0) &&
                      GPU_EXEC ==
                          1) { // If the number of input queue elements is higher than or equal to switching_limit

                if(rep >= p.n_warmup)
                    timer.start("Copy To Device");
                dpct::dpct_memcpy(d_cost, h_cost, sizeof(int) * n_nodes,
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_color, h_color, sizeof(int) * n_nodes,
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_threads_run, h_threads_run, sizeof(int),
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_threads_end, h_threads_end, sizeof(int),
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_overflow, h_overflow, sizeof(int),
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_q1, h_q1, sizeof(int) * n_nodes,
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_q2, h_q2, sizeof(int) * n_nodes,
                                  dpct::host_to_device);
                dpct::dpct_memcpy(d_iter, h_iter, sizeof(int),
                                  dpct::host_to_device);
                dev_ct1.queues_wait_and_throw();
                //CUDA_ERR();
                if(rep >= p.n_warmup)
                    timer.stop("Copy To Device");


                // Continue until switching_limit condition is not satisfied
                while((*h_num_t != 0) && (*h_num_t >= p.switching_limit || CPU_EXEC == 0) && GPU_EXEC == 1) {
                   //printf("h_iter %d\n", h_iter[0].load());

                    // Swap queues
                    if(h_iter[0] % 2 == 0) {
                        d_qin  = d_q1;
                        d_qout = d_q2;
                    } else {
                        d_qin  = d_q2;
                        d_qout = d_q1;
                    }

                    if(rep >= p.n_warmup)
                        timer.start("Copy To Device");
                    dpct::dpct_memcpy(d_num_t, h_num_t, sizeof(int),
                                      dpct::host_to_device);
                    dpct::dpct_memcpy(d_tail, h_tail, sizeof(int),
                                      dpct::host_to_device);
                    dpct::dpct_memcpy(d_head, h_head, sizeof(int),
                                      dpct::host_to_device);
                    dpct::dpct_memcpy(d_gray_shade, h_gray_shade, sizeof(int),
                                      dpct::host_to_device);
                    dev_ct1.queues_wait_and_throw();
                    //CUDA_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Copy To Device");

                    if(rep >= p.n_warmup)
                        timer.start("Kernel");
                    assert(p.n_gpu_threads <= max_gpu_threads && 
                        "The thread block size is greater than the maximum thread block size that can be used on this device");

                    sycl::range<3> dimGrid(p.n_gpu_blocks, 1, 1);
                    sycl::range<3> dimBlock(p.n_gpu_threads, 1, 1);

                    {
                        dpct::buffer_t d_nodes_buf_ct0 =
                            dpct::get_buffer(d_nodes);
                        dpct::buffer_t d_edges_buf_ct1 =
                            dpct::get_buffer(d_edges);
                        dpct::buffer_t d_cost_buf_ct2 =
                            dpct::get_buffer(d_cost);
                        dpct::buffer_t d_color_buf_ct3 =
                            dpct::get_buffer(d_color);
                        std::pair<dpct::buffer_t, size_t> d_qin_buf_ct4 =
                            dpct::get_buffer_and_offset(d_qin);
                        size_t d_qin_offset_ct4 = d_qin_buf_ct4.second;
                        std::pair<dpct::buffer_t, size_t> d_qout_buf_ct5 =
                            dpct::get_buffer_and_offset(d_qout);
                        size_t d_qout_offset_ct5 = d_qout_buf_ct5.second;
                        dpct::buffer_t d_num_t_buf_ct6 =
                            dpct::get_buffer(d_num_t);
                        dpct::buffer_t d_head_buf_ct7 =
                            dpct::get_buffer(d_head);
                        dpct::buffer_t d_tail_buf_ct8 =
                            dpct::get_buffer(d_tail);
                        dpct::buffer_t d_threads_end_buf_ct9 =
                            dpct::get_buffer(d_threads_end);
                        dpct::buffer_t d_threads_run_buf_ct10 =
                            dpct::get_buffer(d_threads_run);
                        dpct::buffer_t d_overflow_buf_ct11 =
                            dpct::get_buffer(d_overflow);
                        dpct::buffer_t d_gray_shade_buf_ct12 =
                            dpct::get_buffer(d_gray_shade);
                        dpct::buffer_t d_iter_buf_ct13 =
                            dpct::get_buffer(d_iter);
                        dpct::get_default_queue().submit([&](sycl::handler
                                                                 &cgh) {
                            sycl::accessor<int, 1,
                                           sycl::access::mode::read_write,
                                           sycl::access::target::local>
                                l_mem_acc_ct1(
                                    sycl::range<1>(1602 /*W_QUEUE_SIZE+2*/),
                                    cgh);
                            sycl::accessor<int, 0,
                                           sycl::access::mode::read_write,
                                           sycl::access::target::local>
                                tail_bin_acc_ct1(cgh);
                            auto d_nodes_acc_ct0 =
                                d_nodes_buf_ct0
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_edges_acc_ct1 =
                                d_edges_buf_ct1
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_cost_acc_ct2 =
                                d_cost_buf_ct2
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_color_acc_ct3 =
                                d_color_buf_ct3
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_qin_acc_ct4 =
                                d_qin_buf_ct4.first
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_qout_acc_ct5 =
                                d_qout_buf_ct5.first
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_num_t_acc_ct6 =
                                d_num_t_buf_ct6
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_head_acc_ct7 =
                                d_head_buf_ct7
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_tail_acc_ct8 =
                                d_tail_buf_ct8
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_threads_end_acc_ct9 =
                                d_threads_end_buf_ct9
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_threads_run_acc_ct10 =
                                d_threads_run_buf_ct10
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_overflow_acc_ct11 =
                                d_overflow_buf_ct11
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_gray_shade_acc_ct12 =
                                d_gray_shade_buf_ct12
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);
                            auto d_iter_acc_ct13 =
                                d_iter_buf_ct13
                                    .get_access<sycl::access::mode::read_write>(
                                        cgh);

                            auto dpct_global_range = dimGrid * dimBlock;

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(dpct_global_range.get(2),
                                                   dpct_global_range.get(1),
                                                   dpct_global_range.get(0)),
                                    sycl::range<3>(dimBlock.get(2),
                                                   dimBlock.get(1),
                                                   dimBlock.get(0))),
                                [=](sycl::nd_item<3> item_ct1) {
                                    int *d_qin_ct4 = (int *)(&d_qin_acc_ct4[0] +
                                                             d_qin_offset_ct4);
                                    int *d_qout_ct5 =
                                        (int *)(&d_qout_acc_ct5[0] +
                                                d_qout_offset_ct5);
                                    SSSP_gpu(
                                        (Node *)(&d_nodes_acc_ct0[0]),
                                        (Edge *)(&d_edges_acc_ct1[0]),
                                        (int *)(&d_cost_acc_ct2[0]),
                                        (int *)(&d_color_acc_ct3[0]), d_qin_ct4,
                                        d_qout_ct5,
                                        (int *)(&d_num_t_acc_ct6[0]),
                                        (int *)(&d_head_acc_ct7[0]),
                                        (int *)(&d_tail_acc_ct8[0]),
                                        (int *)(&d_threads_end_acc_ct9[0]),
                                        (int *)(&d_threads_run_acc_ct10[0]),
                                        (int *)(&d_overflow_acc_ct11[0]),
                                        (int *)(&d_gray_shade_acc_ct12[0]),
                                        (int *)(&d_iter_acc_ct13[0]),
                                        p.switching_limit, CPU_EXEC, item_ct1,
                                        l_mem_acc_ct1.get_pointer(),
                                        tail_bin_acc_ct1.get_pointer());
                                });
                        });
                    }

                    dev_ct1.queues_wait_and_throw();
                    //CUDA_ERR();
                    if(rep >= p.n_warmup)
                        timer.stop("Kernel");

                    if(rep >= p.n_warmup)
                        timer.start("Copy Back and Merge");
                    dpct::dpct_memcpy(h_tail, d_tail, sizeof(int),
                                      dpct::device_to_host);
                    dpct::dpct_memcpy(h_iter, d_iter, sizeof(int),
                                      dpct::device_to_host);
                    dev_ct1.queues_wait_and_throw();
                    //CUDA_ERR();
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
                dpct::dpct_memcpy(h_cost, d_cost, sizeof(int) * n_nodes,
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_color, d_color, sizeof(int) * n_nodes,
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_threads_run, d_threads_run, sizeof(int),
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_threads_end, d_threads_end, sizeof(int),
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_overflow, d_overflow, sizeof(int),
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_q1, d_q1, sizeof(int) * n_nodes,
                                  dpct::device_to_host);
                dpct::dpct_memcpy(h_q2, d_q2, sizeof(int) * n_nodes,
                                  dpct::device_to_host);
                dev_ct1.queues_wait_and_throw();
                //CUDA_ERR();
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
    verify(h_cost, n_nodes, p.comparison_file);

    // Free memory
    timer.start("Deallocation");
    free(h_nodes);
    free(h_edges);
    free(h_color);
    free(h_cost);
    free(h_q1);
    free(h_q2);
    dpct::dpct_free(d_nodes);
    dpct::dpct_free(d_edges);
    dpct::dpct_free(d_cost);
    dpct::dpct_free(d_color);
    dpct::dpct_free(d_q1);
    dpct::dpct_free(d_q2);
    dpct::dpct_free(d_num_t);
    dpct::dpct_free(d_head);
    dpct::dpct_free(d_tail);
    dpct::dpct_free(d_threads_end);
    dpct::dpct_free(d_threads_run);
    dpct::dpct_free(d_overflow);
    dpct::dpct_free(d_iter);
    dpct::dpct_free(d_gray_shade);
    //CUDA_ERR();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);


    printf("Test Passed\n");
    return 0;
}
