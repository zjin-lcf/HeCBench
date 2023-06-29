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

#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <atomic>
#include <chrono>

#include "support/setup.h"
#include "kernel.h"
#include "support/task.h"
#include "support/verify.h"

using namespace std;

// Params ---------------------------------------------------------------------
struct Params {

    int         n_gpu_threads;
    int         n_gpu_blocks;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    const char *file_name;
    int         pattern;
    int         pool_size;
    int         queue_size;
    int         iterations;

    Params(int argc, char **argv) {
        n_gpu_threads = 64;
        n_gpu_blocks  = 320;
        n_threads     = 1;
        n_warmup      = 5;
        n_reps        = 1000;
        file_name     = "input/patternsNP100NB512FB25.txt";
        pattern       = 1;
        pool_size     = 3200;
        queue_size    = 320;
        iterations    = 1000;
        int opt;
        while((opt = getopt(argc, argv, "hi:g:t:w:r:f:k:s:q:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_gpu_threads = atoi(optarg); break;
            case 'g': n_gpu_blocks  = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'f': file_name     = optarg; break;
            case 'k': pattern       = atoi(optarg); break;
            case 's': pool_size     = atoi(optarg); break;
            case 'q': queue_size    = atoi(optarg); break;
            case 'n': iterations    = atoi(optarg); break;
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
                "\nUsage:  ./tq [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device threads per block (default=64)"
                "\n    -g <G>    # of device blocks (default=320)"
                "\n    -t <T>    # of host threads (default=1)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=1000)"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    patterns file name (default=input/patternsNP100NB512FB25.txt)"
                "\n    -k <K>    pattern in file (default=1)"
                "\n    -s <S>    task pool size (default=3200)"
                "\n    -q <Q>    task queue size (default=320)"
                "\n    -n <N>    # of iterations in heavy task (default=1000)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(int *pattern, task_t *task_pool, const Params &p) {

    // Patterns file name
    char filePatterns[100];

    sprintf(filePatterns, "%s", p.file_name);

    // Read line from patterns file
    FILE *File;
    int r;
    if((File = fopen(filePatterns, "rt")) != NULL) {
        for(int y = 0; y <= p.pattern; y++) {
            for(int x = 0; x < 512; x++) {
                fscanf(File, "%d ", &r);
                pattern[x] = r;
            }
        }
        fclose(File);
    } else {
        printf("Unable to open file %s\n", filePatterns);
        exit(-1);
    }

    for(int i = 0; i < p.pool_size; i++) {
        //Setting tasks in the tasks pool
        task_pool[i].id = i;
        task_pool[i].op = SIGNAL_NOTWORK_KERNEL;
    }

    //Read the pattern
    for(int i = 0; i < p.pool_size; i++) {
        pattern[i] = pattern[i%512];
        if(pattern[i] == 1) {
            task_pool[i].op = SIGNAL_WORK_KERNEL;
        }
    }
}

int main(int argc, char **argv) {

    const Params p(argc, argv);

    const int max_gpu_threads = 256; // 1024
    assert(p.n_gpu_threads <= max_gpu_threads && 
           "The thread block size is greater than the maximum thread block size that can be used on this device");

    // Allocate
    int *   h_pattern     = (int *)malloc(p.pool_size * sizeof(int));
    task_t *h_task_pool   = (task_t *)malloc(p.pool_size * sizeof(task_t));
    task_t *h_task_queues = (task_t *)malloc(p.queue_size * sizeof(task_t));

    int *   h_data_pool   = (int *)malloc(p.pool_size * p.n_gpu_threads * sizeof(int));
    int *   h_data_queues = (int *)malloc(p.queue_size * p.n_gpu_threads * sizeof(int));

    int *  h_consumed = (int *)malloc(sizeof(int));

    ALLOC_ERR(h_pattern, h_task_pool, h_task_queues, h_data_pool, h_data_queues, h_consumed);

    // Initialize
    read_input(h_pattern, h_task_pool, p);
    memset((void *)h_data_pool, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
    memset((void *)h_consumed, 0, sizeof(int));

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    task_t * d_task_queues = sycl::malloc_device<task_t>(p.queue_size, q);

    int * d_data_queues = sycl::malloc_device<int>(p.queue_size * p.n_gpu_threads, q);

    int * d_consumed = sycl::malloc_device<int>(1, q);

    auto start = std::chrono::steady_clock::now();

    for(int rep = 0; rep < p.n_reps + p.n_warmup; rep++) {

        // Reset
        memset((void *)h_data_pool, 0, p.pool_size * p.n_gpu_threads * sizeof(int));
        int n_written_tasks = 0;

        for(int n_consumed_tasks = 0; n_consumed_tasks < p.pool_size; n_consumed_tasks += p.queue_size) {

            host_insert_tasks(h_task_queues, h_data_queues, h_task_pool, h_data_pool, &n_written_tasks, p.queue_size,
                              n_consumed_tasks, p.n_gpu_threads);

            q.memcpy(d_task_queues, h_task_queues, p.queue_size * sizeof(task_t));
            q.memcpy(d_data_queues, h_data_queues, p.queue_size * p.n_gpu_threads * sizeof(int));
            q.memcpy(d_consumed, h_consumed, sizeof(int));

            // Kernel launch
            call_TaskQueue_gpu(q, p.n_gpu_blocks, p.n_gpu_threads, d_task_queues, d_data_queues, d_consumed, 
                p.iterations, n_consumed_tasks, p.queue_size, sizeof(int) + sizeof(task_t));

            q.memcpy(&h_data_pool[n_consumed_tasks * p.n_gpu_threads],
                     d_data_queues,
                     p.queue_size * p.n_gpu_threads * sizeof(int)).wait();
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total task execution time for %d iterations: %f (ms)\n", p.n_reps + p.n_warmup, time * 1e-6f);

    sycl::free(d_task_queues, q);
    sycl::free(d_data_queues, q);
    sycl::free(d_consumed, q);

    // Verify answer
    verify(h_data_pool, h_pattern, p.pool_size, p.iterations, p.n_gpu_threads);

    free(h_pattern);
    free(h_consumed);
    free(h_task_queues);
    free(h_data_queues);
    free(h_task_pool);
    free(h_data_pool);

    printf("Test Passed\n");
    return 0;
}
