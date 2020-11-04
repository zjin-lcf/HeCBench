#pragma omp target teams num_teams(done_size) thread_limit(LQSORT_LOCAL_WORKGROUP_SIZE)
{
  workstack_record workstack[QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD]; 
  int workstack_pointer;

  T mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE], temp[SORT_THRESHOLD];
  T *s, *sn;
  uint ltsum, gtsum;
  uint lt[LQSORT_LOCAL_WORKGROUP_SIZE+1], gt[LQSORT_LOCAL_WORKGROUP_SIZE+1];
#pragma omp parallel
  {
    const uint blockid    = omp_get_team_num();
    const uint localid    = omp_get_thread_num();

    // workstack: stores the start and end of the sequences, direction of sort
    // If the sequence is less that SORT_THRESHOLD, it gets sorted. 
    // It will only be pushed on the stack if it greater than the SORT_THRESHOLD. 
    // Note, that the sum of ltsum + gtsum is less than QUICKSORT_BLOCK_SIZE. 
    // The total sum of the length of records on the stack cannot exceed QUICKSORT_BLOCK_SIZE, 
    // but each individual record should be greater than SORT_THRESHOLD, so the maximum length 
    // of the stack is QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD - in the case of BDW GT2 the length 
    // of the stack is 2 :)
    uint i, tmp, ltp, gtp;

    work_record<T> block = seqs[blockid];
    const uint d_offset = block.start;
    uint start = 0; 
    uint end   = block.end - d_offset;

    uint direction = 1; // which direction to sort
    // initialize workstack and workstack_pointer: push the initial sequence on the stack
    if (localid == 0) {
      workstack_pointer = 0; // beginning of the stack
      workstack_record wr = { start, end, direction };
      workstack[0] = wr;
    }
    // copy block of data to be sorted by one workgroup into __shared__ memory
    // note that indeces of __shared__ data go from 0 to end-start-1
    if (block.direction == 1) {
      for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
        mys[i] = d[i+d_offset];
      }
    } else {
      for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
        mys[i] = dn[i+d_offset];
      }
    }
#pragma omp barrier

    while (workstack_pointer >= 0) { 
      // pop up the stack
      workstack_record wr = workstack[workstack_pointer];
      start = wr.start;
      end = wr.end;
      direction = wr.direction;
#pragma omp barrier
      if (localid == 0) {
        --workstack_pointer;

        ltsum = gtsum = 0;	
      }
      if (direction == 1) {
        s = mys;
        sn = mysn;
      } else {
        s = mysn;
        sn = mys;
      }
      // Set thread __shared__ counters to zero
      lt[localid] = gt[localid] = 0;
      ltp = gtp = 0;
#pragma omp barrier

      // Pick a pivot
      uint pivot = s[start];
      if (start < end) {
        pivot = median(pivot, s[(start+end) >> 1], s[end-1]);
      }
      // Align work item accesses for coalesced reads.
      // Go through data...
      for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
        tmp = s[i];
        // counting elements that are smaller ...
        if (tmp < pivot)
          ltp++;
        // or larger compared to the pivot.
        if (tmp > pivot) 
          gtp++;
      }
      lt[localid] = ltp;
      gt[localid] = gtp;
#pragma omp barrier

      // calculate cumulative sums
      uint n;
      for(i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
        n = 2*i - 1;
        if ((localid & n) == n) {
          lt[localid] += lt[localid-i];
          gt[localid] += gt[localid-i];
        }
#pragma omp barrier
      }

      if ((localid & n) == n) {
        lt[LQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
        gt[LQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
        lt[localid] = 0;
        gt[localid] = 0;
      }

      for(i = LQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
        n = 2*i - 1;
        if ((localid & n) == n) {
          plus_prescan(&lt[localid - i], &lt[localid]);
          plus_prescan(&gt[localid - i], &gt[localid]);
        }
#pragma omp barrier
      }

      // Allocate locations for work items
      uint lfrom = start + lt[localid];
      uint gfrom = end - gt[localid+1];

      // go thru data again writing elements to their correct position
      for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
        tmp = s[i];
        // increment counts
        if (tmp < pivot) 
          sn[lfrom++] = tmp;

        if (tmp > pivot) 
          sn[gfrom++] = tmp;
      }
#pragma omp barrier

      // Store the pivot value between the new sequences
      for (i = start + ltsum + localid;i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
        d[i+d_offset] = pivot;
      }
#pragma omp barrier

      // if the sequence is shorter than SORT_THRESHOLD
      // sort it using an alternative sort and place result in d
      if (ltsum <= SORT_THRESHOLD) {
        sort_threshold(sn, d+d_offset, start, start + ltsum, temp, localid);
      } else {
        PUSH(start, start + ltsum);
#pragma omp barrier
      }

      if (gtsum <= SORT_THRESHOLD) {
        sort_threshold(sn, d+d_offset, end - gtsum, end, temp, localid);
      } else {
        PUSH(end - gtsum, end);
#pragma omp barrier
      }
    }
  }
}


