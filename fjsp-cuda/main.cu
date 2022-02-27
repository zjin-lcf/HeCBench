/*
 * Original Author      : imtsuki
 * Copyright   : imtsuki <me@qjx.app>
 * Description : Flexible Job Shop Scheduling Problem
 */

#include <algorithm>  // std::min_element
#include <chrono>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * Check the return value of the CUDA runtime API call
 * and report the error when the call fails
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
    const char *statement, cudaError_t err) {
  if (err != cudaSuccess)
    std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
              << err << ") at " << file << ":" << line << std::endl;
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

const int MAX_OPERATIONS_PER_STEP = 5;
const int MAX_STEPS_PER_JOB = 20;
const int MAX_JOBS = 20;
const int MAX_MACHINES = 20;

int POPULATION_SIZE = 2000;
int INDIVIDUAL_LEN = 20;
const int SIZE_PARENT_POOL = 7;

int total_jobs, total_machines, max_operations;

struct Operation {
  int id_machine;
  int processing_time;
};

struct Step {
  int len;
  Operation candidates[MAX_OPERATIONS_PER_STEP];
};

struct Job {
  int len;
  Step steps[MAX_STEPS_PER_JOB];
};

Job input_data[MAX_JOBS];

struct Gene {
  int id_job;
  int id_step;
  // Make sure update them both.
  int id_machine;
  int id_operation;
};

std::ostream &operator<<(std::ostream &os, const Gene &gene) {
  os << "[" << gene.id_job << ", " << gene.id_step << ", "
     << gene.id_operation << "]";
  return os;
}

void parse_input(const char *path) {
  auto input = std::ifstream();

  input.exceptions(std::ifstream::failbit);

  input.open(path);

  input >> total_jobs >> total_machines >> max_operations;

  if (total_jobs > MAX_JOBS) {
    throw std::runtime_error("Too many jobs");
  }

  if (total_machines > MAX_MACHINES) {
    throw std::runtime_error("Too many machines");
  }

  INDIVIDUAL_LEN = 0;

  for (int id_job = 0; id_job < total_jobs; id_job++) {
    int number_steps;
    input >> number_steps;

    if (number_steps > MAX_STEPS_PER_JOB) {
      throw std::runtime_error("Too many steps");
    }

    input_data[id_job].len = number_steps;

    for (int id_step = 0; id_step < number_steps; id_step++) {
      int number_operations;
      input >> number_operations;

      if (number_operations > MAX_OPERATIONS_PER_STEP) {
        throw std::runtime_error("Too many operations");
      }

      input_data[id_job].steps[id_step].len = number_operations;

      for (int id_operation = 0; id_operation < number_operations;
          id_operation++) {
        int id_machine;
        int processing_time;
        input >> id_machine >> processing_time;
        input_data[id_job].steps[id_step].candidates[id_operation].id_machine =
          id_machine - 1;
        input_data[id_job].steps[id_step].candidates[id_operation].processing_time =
          processing_time;
      }
      INDIVIDUAL_LEN++;
    }
  }
}

__device__ unsigned LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

__device__ void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

__global__ void init_rand_state_kernel(unsigned int* states, int len, unsigned int seed)
{ 
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < len; i += stride) {
    states[i] = seed ^ i;
    LCG_random_init(&states[i]);
  }
}

__global__ void fill_rand_kernel(int *__restrict__ numbers, int len, int max_value, 
                                 unsigned *__restrict__ rand_states) 
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x + threadIdx.x;
  for (int i = index; i < len; i += stride) 
     numbers[i] = LCG_random(&rand_states[i]) % max_value;
}

__global__ void init_population_kernel(
  Gene *__restrict__ population,
  int population_size,
  int individual_len,
  const Job *__restrict__ jobs,
  int total_jobs,
  unsigned *__restrict__ rand_states)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int next_step[MAX_JOBS];

  for (int i = index; i < population_size; i += stride) {
    int cursor = 0;
    Gene *me = population + i * individual_len;
    for (int j = 0; j < MAX_JOBS; j++) next_step[j] = 0;
    while (cursor < individual_len) {
      int id_job = LCG_random(&rand_states[i]) % total_jobs;
      if (next_step[id_job] < jobs[id_job].len) {
        me[cursor].id_job = id_job;
        me[cursor].id_step = next_step[id_job];
        next_step[id_job]++;
        cursor++;
      }
    }
  }
}

__global__ void pick_parents_kernel(
  int *__restrict__ parents,
  const int *__restrict__ parent_candidates,
  const int *__restrict__ scores,
  int population_size)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  if (index < population_size) {
    for (int i = index; i < population_size; i += stride) {
      int best_score = INT_MAX;
      int best_index = -1;

      for (int j = 0; j < SIZE_PARENT_POOL; j++) {
        int k = parent_candidates[i * SIZE_PARENT_POOL + j];
        if (scores[k] < best_score) {
          best_score = scores[k];
          best_index = k;
        }
      }

      parents[i] = best_index;
    }
  }
}

__device__ void assignment_crossover(
        Gene *__restrict__ child,
  const Gene *__restrict__ parent_a,
  const Gene *__restrict__ parent_b,
  int individual_len)
{
  int reverse_index[MAX_JOBS][MAX_STEPS_PER_JOB];

  for (int s = 0; s < individual_len; s++) {
    int id_job = parent_b[s].id_job;
    int id_step = parent_b[s].id_step;
    reverse_index[id_job][id_step] = s;
  }

  for (int s = 0; s < individual_len; s++) {
    int id_job = parent_a[s].id_job;
    int id_step = parent_a[s].id_step;
    int i = reverse_index[id_job][id_step];

    child[s] = parent_a[s];
    child[s].id_operation = parent_b[i].id_operation;
    child[s].id_machine = parent_b[i].id_machine;
  }
}

__device__ void sequencing_crossover(
        Gene *__restrict__ child,
  const Gene *__restrict__ parent_a,
  const Gene *__restrict__ parent_b,
  int individual_len,
  unsigned *rand_state)
{
  int crossover_point = LCG_random(rand_state) % individual_len;

  int last_step[MAX_JOBS];

  for (int i = 0; i < MAX_JOBS; i++) {
    last_step[i] = -1;
  }

  for (int s = 0; s < crossover_point; s++) {
    int id_job = parent_b[s].id_job;
    int id_step = parent_b[s].id_step;

    child[s] = parent_b[s];
    last_step[id_job] = id_step;
  }

  int cursor = crossover_point;

  for (int s = 0; s < individual_len; s++) {
    int id_job = parent_a[s].id_job;

    if (last_step[id_job] < parent_a[s].id_step) {
      child[cursor] = parent_a[s];
      cursor++;
    }
  }
}

__device__ void assignment_mutation(
  Gene *__restrict__ individual,
  int individual_len,
  Job *__restrict__ jobs,
  unsigned *rand_state)
{
  int count = 5;
  while (count--) {
    int mutation_point = LCG_random(rand_state) % individual_len;
    int id_job = individual[mutation_point].id_job;
    int id_step = individual[mutation_point].id_step;
    int len = jobs[id_job].steps[id_step].len;
    int id_operation = LCG_random(rand_state) % len;

    individual[mutation_point].id_operation = id_operation;
    individual[mutation_point].id_machine =
      jobs[id_job].steps[id_step].candidates[id_operation].id_machine;
  }
}

__device__ void swapping_mutation(
  Gene *__restrict__ individual,
  int individual_len,
  Job *__restrict__ jobs,
  unsigned *rand_state)
{
  int count = 5;
  while (count--) {
    int mutation_point = LCG_random(rand_state) % (individual_len - 1);

    if (individual[mutation_point].id_job != individual[mutation_point + 1].id_job) {
       Gene t = individual[mutation_point];
       individual[mutation_point] = individual[mutation_point + 1];
       individual[mutation_point + 1] = t;
    }
  }
}

__global__ void stage_1_breed_kernel(
  const int *__restrict__ parents,
  const Gene *__restrict__ population,
  Gene *__restrict__ new_population,
  int population_size,
  int individual_len,
  Job *__restrict__ jobs,
  unsigned *__restrict__ rand_states)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < population_size; i += stride) {
    if (i < population_size * 8 / 10) {
      sequencing_crossover(&new_population[i * individual_len],
          &population[parents[i] * individual_len],
          &population[parents[i + 1] * individual_len],
          individual_len, &rand_states[i]);
    } else {
      for (int s = 0; s < individual_len; s++) {
        new_population[i * individual_len + s] =
          population[parents[i] * individual_len + s];
      }

      swapping_mutation(&new_population[i * individual_len],
          individual_len, jobs, &rand_states[i]);
    }
  }
}

__global__ void stage_2_breed_kernel(
  const int *__restrict__ parents,
  const Gene *__restrict__ population,
  Gene *__restrict__ new_population,
  int population_size,
  int individual_len,
  Job *__restrict__ jobs,
  unsigned *__restrict__ rand_states)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < population_size; i += stride) {
    if (i < population_size * 4 / 10) {
      assignment_crossover(&new_population[i * individual_len],
          &population[parents[i] * individual_len],
          &population[parents[i + 1] * individual_len],
          individual_len);
    } else if (i < population_size * 8 / 10) {
      sequencing_crossover(&new_population[i * individual_len],
          &population[parents[i] * individual_len],
          &population[parents[i + 1] * individual_len],
          individual_len, &rand_states[i]);
    } else {
      for (int s = 0; s < individual_len; s++) {
        new_population[i * individual_len + s] =
          population[parents[i] * individual_len + s];
      }
      if (i < population_size * 9 / 10) {
        assignment_mutation(&new_population[i * individual_len],
            individual_len, jobs, &rand_states[i]);
      } else {
        swapping_mutation(&new_population[i * individual_len],
            individual_len, jobs, &rand_states[i]);
      }
    }
  }
}

__global__ void stage_1_evaluate_kernel(
  int *__restrict__ scores,
  Gene *__restrict__ population,
  int population_size,
  int individual_len,
  Job *__restrict__ jobs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int value;
  int machines[MAX_MACHINES];
  int last_step_id_machine[MAX_JOBS];

  for (int i = index; i < population_size; i += stride) {
    value = 0;
    for (int m = 0; m < MAX_MACHINES; m++) machines[m] = 0;
    Gene *me = population + i * individual_len;

    for (int s = 0; s < individual_len; s++) {
      int id_job = me[s].id_job;
      int id_step = me[s].id_step;
      int len = jobs[id_job].steps[id_step].len;
      int best_end_time = INT_MAX;
      int best_id_operation = -1;
      int best_id_machine = -1;

      // Greedy search to find best operation in this step
      for (int id_operation = 0; id_operation < len; id_operation++) {
        int processing_time =
          jobs[id_job].steps[id_step].candidates[id_operation].processing_time;
        int id_machine =
          jobs[id_job].steps[id_step].candidates[id_operation].id_machine;

        int machine_end_time = machines[id_machine];

        if (id_step > 0) {
          int previous_id_machine = last_step_id_machine[id_job];
          if (machine_end_time < machines[previous_id_machine]) {
            machine_end_time = machines[previous_id_machine];
          }
        }

        machine_end_time += processing_time;

        if (machine_end_time < best_end_time) {
          best_end_time = machine_end_time;
          best_id_operation = id_operation;
          best_id_machine = id_machine;
        }
      }
      me[s].id_operation = best_id_operation;
      me[s].id_machine = best_id_machine;
      machines[best_id_machine] = best_end_time;
      last_step_id_machine[id_job] = best_id_machine;
      if (best_end_time > value) {
        value = best_end_time;
      }
    }

    scores[i] = value;
  }
}

__global__ void stage_2_evaluate_kernel(
  int *__restrict__ scores,
  const Gene *__restrict__ population,
  int population_size,
  int individual_len,
  Job *__restrict__ jobs)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int value;
  int machines[MAX_MACHINES];
  int last_step_id_machine[MAX_JOBS];

  for (int i = index; i < population_size; i += stride) {
    value = 0;
    for (int m = 0; m < MAX_MACHINES; m++) machines[m] = 0;
    const Gene *me = population + i * individual_len;

    for (int s = 0; s < individual_len; s++) {
      int id_job = me[s].id_job;
      int id_step = me[s].id_step;
      int id_machine = me[s].id_machine;
      int id_operation = me[s].id_operation;

      int processing_time =
        jobs[id_job].steps[id_step].candidates[id_operation].processing_time;

      int previous_id_machine = last_step_id_machine[id_job];

      if (id_step > 0 && machines[id_machine] < machines[previous_id_machine]) 
        machines[id_machine] = machines[previous_id_machine];

      machines[id_machine] += processing_time;

      if (machines[id_machine] > value) value = machines[id_machine];

      last_step_id_machine[id_job] = id_machine;
    }

    scores[i] = value;
  }
  
}

int main(int argc, const char *argv[]) {
  const char *path = "./data/mk01.fjs";
  if (argc >= 2) path = argv[1];
  parse_input(path);

  std::cout << "total_jobs: " << total_jobs << "\n";
  std::cout << "total_machines: " << total_machines << "\n";
  std::cout << "INDIVIDUAL_LEN: " << INDIVIDUAL_LEN << "\n";

  std::cout << "Print input data:\n";

  for (int id_job = 0; id_job < total_jobs; id_job++) {
    std::cout << "[Job " << id_job << "] ";
    for (int id_step = 0; id_step < input_data[id_job].len; id_step++) {
      std::cout << id_step << ": ";
      for (int id_operation = 0;
          id_operation < input_data[id_job].steps[id_step].len;
          id_operation++) {
        std::cout << "("
          << input_data[id_job].steps[id_step].candidates[id_operation].id_machine
          << ", "
          << input_data[id_job].steps[id_step].candidates[id_operation].processing_time
          << ") ";
      }
    }
    std::cout << "\n";
  }

  // save device results
  std::vector<Gene> population_h(POPULATION_SIZE * INDIVIDUAL_LEN);
  std::vector<int> scores_h(POPULATION_SIZE);

  auto start = std::chrono::high_resolution_clock::now();

  Job *jobs;
  CUDA_CHECK_RETURN(cudaMalloc((void ** )&jobs, MAX_JOBS * sizeof(Job)));
  CUDA_CHECK_RETURN(cudaMemcpy(jobs, input_data, MAX_JOBS * sizeof(Job), cudaMemcpyHostToDevice));

  Gene *population;
  cudaMalloc((void**)&population, sizeof(Gene) * POPULATION_SIZE * INDIVIDUAL_LEN);

  int *scores;
  cudaMalloc((void**)&scores, sizeof(int) * POPULATION_SIZE);

  Gene *new_population;
  cudaMalloc((void**)&new_population, sizeof(Gene) * POPULATION_SIZE * INDIVIDUAL_LEN);

  // Parent candidate indexes
  int *parent_candidates;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&parent_candidates, POPULATION_SIZE * SIZE_PARENT_POOL * sizeof(int)));

  // Picked parent indexes
  int *parents;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&parents, POPULATION_SIZE * sizeof(int)));

  // random states
  unsigned *parent_candidates_states;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&parent_candidates_states,
                    POPULATION_SIZE * SIZE_PARENT_POOL * sizeof(unsigned)));

  unsigned *population_states;
  CUDA_CHECK_RETURN(cudaMalloc((void**)&population_states, POPULATION_SIZE * sizeof(unsigned)));

  // initialize states for generating random numbers
  const int BLOCKS = 2048;
  const int BLOCK_SIZE = 256;

  init_rand_state_kernel<<<BLOCKS, BLOCK_SIZE>>>(population_states, POPULATION_SIZE, 5551212);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

  init_rand_state_kernel<<<BLOCKS, BLOCK_SIZE>>>(parent_candidates_states, POPULATION_SIZE * SIZE_PARENT_POOL, 1212555);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

  init_population_kernel<<<BLOCKS, BLOCK_SIZE>>>(population,
      POPULATION_SIZE, INDIVIDUAL_LEN, jobs, total_jobs, population_states);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

  stage_1_evaluate_kernel<<<BLOCKS, BLOCK_SIZE>>>(scores, population, POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
  CUDA_CHECK_RETURN(cudaPeekAtLastError());

  int stage_1 = 3000;

  while (stage_1--) {

    fill_rand_kernel<<<BLOCKS, BLOCK_SIZE>>>(
        parent_candidates, POPULATION_SIZE * SIZE_PARENT_POOL,
        POPULATION_SIZE, parent_candidates_states);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    pick_parents_kernel<<<BLOCKS, BLOCK_SIZE>>>(parents,
        parent_candidates, scores, POPULATION_SIZE);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    stage_1_breed_kernel<<<BLOCKS, BLOCK_SIZE>>>(parents, population,
        new_population, POPULATION_SIZE, INDIVIDUAL_LEN, jobs,
        population_states);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    cudaMemcpy(population, new_population, sizeof(Gene) * POPULATION_SIZE * INDIVIDUAL_LEN, cudaMemcpyDeviceToDevice); 

    stage_1_evaluate_kernel<<<BLOCKS, BLOCK_SIZE>>>(scores,
        population, POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
  }

  int stage_2 = 2000;

  while (stage_2--) {
    fill_rand_kernel<<<BLOCKS, BLOCK_SIZE>>>(
      parent_candidates, POPULATION_SIZE * SIZE_PARENT_POOL,
      POPULATION_SIZE, parent_candidates_states);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    pick_parents_kernel<<<BLOCKS, BLOCK_SIZE>>>(
      parents, parent_candidates, scores, POPULATION_SIZE);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    stage_2_breed_kernel<<<BLOCKS, BLOCK_SIZE>>>(
      parents, population, new_population, POPULATION_SIZE, 
      INDIVIDUAL_LEN, jobs, population_states);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());

    cudaMemcpy(population, new_population, sizeof(Gene) * POPULATION_SIZE * INDIVIDUAL_LEN, cudaMemcpyDeviceToDevice); 

    stage_2_evaluate_kernel<<<BLOCKS, BLOCK_SIZE>>>(scores,
        population, POPULATION_SIZE, INDIVIDUAL_LEN, jobs);
    CUDA_CHECK_RETURN(cudaPeekAtLastError());
  }

  cudaMemcpy(scores_h.data(), scores, sizeof(int) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(population_h.data(), population, sizeof(Gene) * POPULATION_SIZE, cudaMemcpyDeviceToHost);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float> time = end - start;
  std::cout << "Total execution time: " << time.count() << std::endl;
 
  auto min_iter = std::min_element(scores_h.begin(), scores_h.end());

  int index = min_iter - scores_h.begin();

  std::cout << "Done" << std::endl;

  std::cout << "Solution score: " << scores_h[index] << std::endl;

  for (int i = 0; i < INDIVIDUAL_LEN; i++)
    std::cout << population_h[index * INDIVIDUAL_LEN + i] << " ";
  std::cout << std::endl;

  CUDA_CHECK_RETURN(cudaFree(new_population));
  CUDA_CHECK_RETURN(cudaFree(population));
  CUDA_CHECK_RETURN(cudaFree(scores));
  CUDA_CHECK_RETURN(cudaFree(parent_candidates_states));
  CUDA_CHECK_RETURN(cudaFree(population_states));
  CUDA_CHECK_RETURN(cudaFree(parent_candidates));
  CUDA_CHECK_RETURN(cudaFree(parents));
  CUDA_CHECK_RETURN(cudaFree(jobs));

  return 0;
}

