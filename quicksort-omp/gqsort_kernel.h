#pragma omp target teams num_teams(blocks_size) thread_limit(GQSORT_LOCAL_WORKGROUP_SIZE)
{
  uint lt[GQSORT_LOCAL_WORKGROUP_SIZE+1];
  uint gt[GQSORT_LOCAL_WORKGROUP_SIZE+1];
  uint ltsum, gtsum, lbeg, gbeg;
#pragma omp parallel
  {
    const uint blockid = omp_get_team_num();
    const uint localid = omp_get_thread_num();

    uint i, lfrom, gfrom, ltp = 0, gtp = 0;
    T lpivot, gpivot, tmp; 
    T *s, *sn;

    // Get the sequence block assigned to this work group
    block_record<T> block = blocksb[blockid];
    uint start = block.start, end = block.end, pivot = block.pivot, direction = block.direction;

    parent_record* pparent = parentsb + block.parent; 
    uint* psstart, *psend, *poldstart, *poldend, *pblockcount;

    // GPU-Quicksort cannot sort in place, as the regular quicksort algorithm can.
    // It therefore needs two arrays to sort things out. We start sorting in the 
    // direction of d -> dn and then change direction after each run of gqsort_kernel.
    // Which direction we are sorting: d -> dn or dn -> d?
    if (direction == 1) {
      s = d;
      sn = dn;
    } else {
      s = dn;
      sn = d;
    }

    // Set thread __shared__ counters to zero
    lt[localid] = gt[localid] = 0;
#pragma omp barrier

    // Align thread accesses for coalesced reads.
    // Go through data...
    for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
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
    for(i = 1; i < GQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
      n = 2*i - 1;
      if ((localid & n) == n) {
        lt[localid] += lt[localid-i];
        gt[localid] += gt[localid-i];
      }
#pragma omp barrier
    }

    if ((localid & n) == n) {
      lt[GQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
      gt[GQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
      lt[localid] = 0;
      gt[localid] = 0;
    }

    for(i = GQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
      n = 2*i - 1;
      if ((localid & n) == n) {
        plus_prescan(&lt[localid - i], &lt[localid]);
        plus_prescan(&gt[localid - i], &gt[localid]);
      }
#pragma omp barrier
    }

    // Allocate memory in the sequence this block is a part of
    if (localid == 0) {
      // get shared variables
      psstart = &pparent->sstart;
      psend = &pparent->send;
      poldstart = &pparent->oldstart;
      poldend = &pparent->oldend;
      pblockcount = &pparent->blockcount;
      // Atomic increment allocates memory to write to.
#pragma omp atomic capture
      {
        lbeg = *psstart;
        *psstart += ltsum;
      }

#pragma omp atomic capture
      {
        gbeg = *psend;
        *psend -= gtsum;
      }
      gbeg -= gtsum;

      //lbeg = atomicAdd(psstart, ltsum);
      // Atomic is necessary since multiple blocks access this
      //gbeg = atomicSub(psend, gtsum) - gtsum;
    }
#pragma omp barrier

    // Allocate locations for work items
    lfrom = lbeg + lt[localid];
    gfrom = gbeg + gt[localid];

    // go thru data again writing elements to their correct position
    for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
      tmp = s[i];
      // increment counts
      if (tmp < pivot) 
        sn[lfrom++] = tmp;

      if (tmp > pivot) 
        sn[gfrom++] = tmp;
    }
#pragma omp barrier

    if (localid == 0) {
      uint old_blockcount;
#pragma omp atomic capture
      {
        old_blockcount = *pblockcount;
        (*pblockcount)--;
      }

      if (old_blockcount == 0) { //if (atomicSub(pblockcount, 1) == 0) 
        uint sstart = *psstart;
        uint send = *psend;
        uint oldstart = *poldstart;
        uint oldend = *poldend;

        // Store the pivot value between the new sequences
        for(i = sstart; i < send; i ++) {
          d[i] = pivot;
        }

        lpivot = sn[oldstart];
        gpivot = sn[oldend-1];
        if (oldstart < sstart) {
          lpivot = median(lpivot,sn[(oldstart+sstart) >> 1], sn[sstart-1]);
        } 
        if (send < oldend) {
          gpivot = median(sn[send],sn[(oldend+send) >> 1], gpivot);
        }

        work_record<T>* result1 = result + 2*blockid;
        work_record<T>* result2 = result1 + 1;

        // change the direction of the sort.
        direction ^= 1;

        work_record<T> r1 = {oldstart, sstart, lpivot, direction};
        *result1 = r1;

        work_record<T> r2 = {send, oldend, gpivot, direction};
        *result2 = r2;
      }
    }
  }
}
