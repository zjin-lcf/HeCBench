    s_seed = 0; 
	  #pragma omp barrier

    int lid = omp_get_thread_num();
int local_range = omp_get_num_threads();

    // Decide if this is the last thread that needs to
    // propagate the seed value
    int last_thread = (lid < num_work_groups &&
                      (lid+1) == num_work_groups) ? 1 : 0;

    for (int d = 0; d < 16; d++)
    {
        T val = 0;
        // Load each block's count for digit d
        if (lid < num_work_groups)
        {
            val = isums[(num_work_groups * d) + lid];
        }
        // Exclusive scan the counts in local memory
        //FPTYPE res = scanLocalMem(val, lmem, 1);
            int idx = lid;
            lmem[idx] = 0;
            idx += local_range;
            lmem[idx] = val;
            #pragma omp barrier
            T t;
            for (int i = 1; i < local_range; i *= 2)
            {
                t = lmem[idx -  i]; 
                #pragma omp barrier
                lmem[idx] += t;     
                #pragma omp barrier
            }
            T res = lmem[idx-1];

        // Write scanned value out to global
        if (lid < num_work_groups)
        {
            isums[(num_work_groups * d) + lid] = res + s_seed;
        }
	      #pragma omp barrier

        if (last_thread)
        {
            s_seed += res + val;
        }
        #pragma omp barrier
    }

