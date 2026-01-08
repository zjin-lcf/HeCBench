#pragma omp declare target
unsigned int LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return *seed;
}

void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}
#pragma omp end declare target

// Maximum iterations to replace do-while loop
#define MAX_ITERATIONS 10000

void FSMKernel(
  const int length,
  const unsigned short *__restrict data,
  int *__restrict best,
  unsigned int *__restrict rndstate,
  unsigned char *__restrict bfsm,
  unsigned char *__restrict same,
  int *__restrict smax,
  int *__restrict sbest,
  int *__restrict oldmax)
{
  #pragma omp target teams num_teams(POPCNT) thread_limit(POPSIZE)
  {
    unsigned char next[FSMSIZE * 2 * POPSIZE];
    int bid = omp_get_team_num();

    #pragma omp parallel
    {
      int i, d, pc, s, bit, id, misses, rnd;
      unsigned char *fsm, state[TABSIZE];

      int lid = omp_get_thread_num();
      fsm = &next[lid * (FSMSIZE * 2)];

      // Initialize shared state (only first thread in team)
      if (lid == 0) {
        oldmax[bid] = 0;
        same[bid] = 0;
      }
      // Removed flush - atomics provide memory ordering

      id = lid + bid * POPSIZE;
      rndstate[id] = SEED ^ id;
      LCG_random_init(&rndstate[id]);

      // initial population
      for (i = 0; i < FSMSIZE * 2; i++) {
        fsm[i] = LCG_random(rndstate+id) & (FSMSIZE - 1);
      }

      // Replace do-while with for loop with max iterations
      for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        // Check convergence condition
        int local_same;
        #pragma omp atomic read
        local_same = same[bid];
        if (local_same >= CUTOFF) break;

        // reset miss counter and initial state
        for (i = 0; i < TABSIZE; i++) state[i] = 0;
        misses = 0;

        // evaluate FSM
        for (i = 0; i < length; i++) {
          d = (int)data[i];
          pc = (d >> 1) & (TABSIZE - 1);
          bit = d & 1;
          s = (int)state[pc];
          misses += bit ^ (s & 1);
          state[pc] = fsm[s + s + bit];
        }

        // determine best FSM
        if (lid == 0) {
          #pragma omp atomic update
          best[2]++;  // increment generation count
          smax[bid] = 0;
          sbest[bid] = 0;
        }

        // Replace critical with atomic operations
        int temp_score = length - misses;
        int old_max;
        #pragma omp atomic read
        old_max = smax[bid];
        if (temp_score > old_max) {
          #pragma omp atomic write
          smax[bid] = temp_score;
        }

        // Update best thread ID
        #pragma omp atomic read
        temp_score = smax[bid];
        if (length - misses == temp_score) {
          int old_best;
          #pragma omp atomic read
          old_best = sbest[bid];
          if (lid > old_best) {
            #pragma omp atomic write
            sbest[bid] = lid;
          }
        }

        bit = 0;
        int local_sbest;
        #pragma omp atomic read
        local_sbest = sbest[bid];

        if (local_sbest == lid) {
          // check if there was an improvement
          int local_oldmax, local_smax;
          #pragma omp atomic read
          local_oldmax = oldmax[bid];
          #pragma omp atomic read
          local_smax = smax[bid];

          #pragma omp atomic update
          same[bid]++;

          if (local_oldmax < local_smax) {
            #pragma omp atomic write
            oldmax[bid] = local_smax;
            #pragma omp atomic write
            same[bid] = 0;
          }
        } else {
          // select 1/8 of threads for mutation
          if ((LCG_random(rndstate+id) & 7) == 0) bit = 1;
        }

        if (bit) {
          // mutate best FSM
          for (i = 0; i < FSMSIZE * 2; i++) {
            rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (next[i + local_sbest * FSMSIZE * 2] ^ rnd) & (FSMSIZE - 1);
          }
        } else {
          // crossover
          for (i = 0; i < FSMSIZE * 2; i++) {
            rnd = LCG_random(rndstate+id) & LCG_random(rndstate+id);
            fsm[i] = (fsm[i] & rnd) | (next[i + local_sbest * FSMSIZE * 2] & ~rnd);
          }
        }
      }

      // record best result of this block
      int local_sbest_final;
      #pragma omp atomic read
      local_sbest_final = sbest[bid];

      if (local_sbest_final == lid) {
        id = bid;
        int score = length - misses;

        // Update best score
        int old_best;
        #pragma omp atomic read
        old_best = best[0];

        if (score > old_best) {
          #pragma omp atomic write
          best[0] = score;
          #pragma omp atomic write
          best[1] = bid;
        }

        // Copy FSM
        for (i = 0; i < FSMSIZE * 2; i++) {
          bfsm[id * (FSMSIZE*2) + i] = fsm[i];
        }
      }
    }
  }
}

void MaxKernel(
  int *__restrict best,
  const unsigned char *__restrict bfsm)
{
  // copy best FSM state assignment over
  int block_id = best[1];

  for (int i = 0; i < FSMSIZE * 2; i++) {
    best[i + 3] = bfsm[block_id * (FSMSIZE * 2) + i];
  }
}
