#include "kernel.hpp"

inline short
warpReduceMax_with_index(short val, short& myIndex, short& myIndex2,
                         int lengthSeqB, bool inverse, sycl::nd_item<1> &item)
{
  short myMax    = val;
  short newInd   = 0;
  short newInd2  = 0;
  short ind      = myIndex;
  short ind2     = myIndex2;

  auto sg = item.get_sub_group();
  int warpSize = sg.get_local_range()[0];
#ifdef INTEL_GPU
  int laneId = sg.get_local_id(); 
#endif

  int rem = sycl::min(lengthSeqB, warpSize);

  for(int offset = rem/2; rem > 0; offset = sycl::max(1, rem/2))
  {
    rem -= offset;
    short tempVal = sg.shuffle_down(val, offset);
    newInd  = sg.shuffle_down(ind, offset);
    newInd2 = sg.shuffle_down(ind2, offset);

    // all shuffles are done
    sg.barrier();
    
    val = sycl::max(val, tempVal);

#ifdef INTEL_GPU
    if(laneId + offset >= warpSize)
    {
      val = 0
      newInd = 0;
      newInd2 = 0;
    }
#endif

    if(val > myMax)
    {
      ind   = newInd;
      ind2  = newInd2;
      myMax = val;
    }

    else if(val == tempVal) // this is kind of redundant and has been done purely to match the results
      // with SSW to get the smallest alignment with highest score. Theoreticaly
      // all the alignmnts with same score are same.
    {
      if (inverse) {
        if(newInd2 > ind2) {
          ind = newInd;
          ind2 = newInd2;
        }
      } else {
        if(newInd < ind) {
          ind = newInd;
          ind2 = newInd2;
        }
      }
    }
  }
  myIndex  = ind;
  myIndex2 = ind2;
  val      = myMax;
  return val;
}

short blockShuffleReduce_with_index(short myVal, short& myIndex, short& myIndex2, int lengthSeqB, 
    short* locTots, short* locInds, short* locInds2, bool inverse, sycl::nd_item<1> &item)
{
  auto sg = item.get_sub_group();
  auto gp = item.get_group();

  short laneId = sg.get_local_id();
  short warpId = sg.get_group_id();
  int threadId = item.get_local_linear_id();

  short myInd  = myIndex;
  short myInd2 = myIndex2;
  myVal = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB, inverse, item);

  if(laneId == 0)
    locTots[warpId] = myVal;
  if(laneId == 0)
    locInds[warpId] = myInd;
  if(laneId == 0)
    locInds2[warpId] = myInd2;

  group_barrier(gp);

  // number of subgroups in a work-group
  int nblocks = sg.get_group_range().size();

  if(threadId < nblocks)
  {
    myVal  = locTots[threadId];
    myInd  = locInds[threadId];
    myInd2 = locInds2[threadId];
  }
  else
  {
    myVal  = 0;
    myInd  = -1;
    myInd2 = -1;
  }
  group_barrier(gp);

  if(warpId == 0)
  {
    myVal    = warpReduceMax_with_index(myVal, myInd, myInd2, lengthSeqB, inverse, item);
    myIndex  = myInd;
    myIndex2 = myInd2;
  }

  return myVal;
}

short findMaxFour(short first, short second, short third, short fourth)
{
  short maxScore = 0;

  maxScore = sycl::max(first,second);
  maxScore = sycl::max(maxScore, third);
  maxScore = sycl::max(maxScore, fourth);

  return maxScore;
}

void sequence_aa_kernel(
    const char*__restrict seqA_array,
    const char*__restrict seqB_array,
    const int*__restrict prefix_lengthA,
    const int*__restrict prefix_lengthB,
          short*__restrict seqA_align_begin,
          short*__restrict seqA_align_end,
          short*__restrict seqB_align_begin,
          short*__restrict seqB_align_end,
          short*__restrict top_scores,
    const short startGap,
    const short extendGap,
    const short*__restrict scoring_matrix,
    const short*__restrict encoding_matrix,
    char*__restrict is_valid,
    short*__restrict sh_prev_E,
    short*__restrict sh_prev_H,
    short*__restrict sh_prev_prev_H,
    short*__restrict local_spill_prev_E,
    short*__restrict local_spill_prev_H,
    short*__restrict local_spill_prev_prev_H,
    short*__restrict sh_aa_encoding,
    short*__restrict sh_aa_scoring,
    short*__restrict locTots,
    short*__restrict locInds,
    short*__restrict locInds2,
    bool   inverse,
    sycl::nd_item<1> &item)

{
  auto sg = item.get_sub_group();
  auto gp = item.get_group();
  int block_Id  = item.get_group_linear_id();
  int thread_Id = item.get_local_linear_id();
  short laneId = sg.get_local_id();
  short warpId = sg.get_group_id();
  int warpSize = sg.get_local_range()[0];

  int lengthSeqA;
  int lengthSeqB;

  // local pointers
  const char* seqA;
  const char* seqB;
  const char* longer_seq;

  // setting up block local sequences and their lengths.
  if(block_Id == 0) {
    lengthSeqA = prefix_lengthA[0];
    lengthSeqB = prefix_lengthB[0];
    seqA       = seqA_array;
    seqB       = seqB_array;
  }
  else
  {
    lengthSeqA = prefix_lengthA[block_Id] - prefix_lengthA[block_Id - 1];
    lengthSeqB = prefix_lengthB[block_Id] - prefix_lengthB[block_Id - 1];
    seqA       = seqA_array + prefix_lengthA[block_Id - 1];
    seqB       = seqB_array + prefix_lengthB[block_Id - 1];
  }

  if(inverse) {
    lengthSeqA = seqA_align_end[block_Id];
    lengthSeqB = seqB_align_end[block_Id];
  }

  // what is the max length and what is the min length
  int maxSize = lengthSeqA > lengthSeqB ? lengthSeqA : lengthSeqB;
  int minSize = lengthSeqA < lengthSeqB ? lengthSeqA : lengthSeqB;

  // shared memory space for storing longer of the two strings
  for(int p = thread_Id; p < minSize; p+=warpSize)
    is_valid[p] = 0;

  is_valid += minSize;

  for(int p = thread_Id; p < minSize; p+=warpSize)
    is_valid[p] = 1;

  is_valid += minSize;
  for(int p = thread_Id; p < minSize; p+=warpSize)
    is_valid[p] = 0;


  char myColumnChar;
  // the shorter of the two strings is stored in thread registers
  if(lengthSeqA < lengthSeqB)
  {
    if(thread_Id < lengthSeqA) {
      myColumnChar = inverse ? seqA[(lengthSeqA - 1) - thread_Id] : seqA[thread_Id];  // read only once
      longer_seq = seqB;
    }
  }
  else
  {
    if(thread_Id < lengthSeqB) {
      myColumnChar = inverse ? seqB[(lengthSeqB - 1) - thread_Id] : seqB[thread_Id];
      longer_seq = seqA;
    }
  }

  // this is required here so that complete sequence has been copied to shared memory
  group_barrier(gp);

  int   i            = 1;
  short thread_max   = 0; // to maintain the thread max score
  short thread_max_i = 0; // to maintain the DP coordinate i for the longer string
  short thread_max_j = 0;// to maintain the DP cooirdinate j for the shorter string

  //initializing registers for storing diagonal values for three recent most diagonals (separate tables for
  //H, E and F)
  short _curr_H = 0, _curr_F = 0, _curr_E = 0;
  short _prev_H = 0, _prev_F = 0, _prev_E = 0;
  short _prev_prev_H = 0, _prev_prev_F = 0, _prev_prev_E = 0;
  short _temp_Val = 0;


  int max_threads = item.get_local_range(0);
  for(int p = thread_Id; p < SCORE_MAT_SIZE; p+=max_threads){
    sh_aa_scoring[p] = scoring_matrix[p];
  }
  for(int p = thread_Id; p < ENCOD_MAT_SIZE; p+=max_threads){
    sh_aa_encoding[p] = encoding_matrix[p];
  }

  // to make sure all shmem allocations have been initialized
  group_barrier(gp);

  for(int diag = 0; diag < lengthSeqA + lengthSeqB - 1; diag++)
  {  // iterate for the number of anti-diagonals

    is_valid = is_valid - (diag < minSize || diag >= maxSize); //move the pointer to left by 1 if cnd true

    _temp_Val = _prev_H; // value exchange happens here to setup registers for next iteration
    _prev_H = _curr_H;
    _curr_H = _prev_prev_H;
    _prev_prev_H = _temp_Val;
    _curr_H = 0;

    _temp_Val = _prev_E;
    _prev_E = _curr_E;
    _curr_E = _prev_prev_E;
    _prev_prev_E = _temp_Val;
    _curr_E = 0;

    _temp_Val = _prev_F;
    _prev_F = _curr_F;
    _curr_F = _prev_prev_F;
    _prev_prev_F = _temp_Val;
    _curr_F = 0;


    if(laneId == warpSize - 1)
    { // if you are the last thread in your warp then spill your values to shmem
      sh_prev_E[warpId] = _prev_E;
      sh_prev_H[warpId] = _prev_H;
      sh_prev_prev_H[warpId] = _prev_prev_H;
    }

    if(diag >= maxSize)
    { // if you are invalid in this iteration, spill your values to shmem
      local_spill_prev_E[thread_Id] = _prev_E;
      local_spill_prev_H[thread_Id] = _prev_H;
      local_spill_prev_prev_H[thread_Id] = _prev_prev_H;
    }

    // this is needed so that all the shmem writes are completed.
    group_barrier(gp);

    if(is_valid[thread_Id] && thread_Id < minSize)
    {
      short fVal = _prev_F + extendGap;
      short hfVal = _prev_H + startGap;
      short valeShfl = sg.shuffle(_prev_E, laneId- 1);
      short valheShfl = sg.shuffle(_prev_H, laneId - 1);

      short eVal=0, heVal = 0;

      if(diag >= maxSize) // when the previous thread has phased out, get value from shmem
      {
        eVal = local_spill_prev_E[thread_Id - 1] + extendGap;
        heVal = local_spill_prev_H[thread_Id - 1]+ startGap;
      }
      else
      {
        eVal =((warpId !=0 && laneId == 0)?sh_prev_E[warpId-1]: valeShfl) + extendGap;
        heVal =((warpId !=0 && laneId == 0)?sh_prev_H[warpId-1]:valheShfl) + startGap;
      }

      if(warpId == 0 && laneId == 0) // make sure that values for lane 0 in warp 0 is not undefined
      {
        eVal = 0;
        heVal = 0;
      }
      _curr_F = (fVal > hfVal) ? fVal : hfVal;
      _curr_E = (eVal > heVal) ? eVal : heVal;

      short testShufll = sg.shuffle(_prev_prev_H, laneId - 1);
      short final_prev_prev_H = 0;
      if(diag >= maxSize)
      {
        final_prev_prev_H = local_spill_prev_prev_H[thread_Id - 1];
      }
      else
      {
        final_prev_prev_H =(warpId !=0 && laneId == 0)?sh_prev_prev_H[warpId-1]:testShufll;
      }


      if(warpId == 0 && laneId == 0) final_prev_prev_H = 0;

      char to_comp = inverse ? longer_seq[maxSize -i] : longer_seq[i - 1];
      short mat_index_q = sh_aa_encoding[(int)to_comp]; //encoding_matrix
      short mat_index_r = sh_aa_encoding[(int)myColumnChar];

      short add_score = sh_aa_scoring[mat_index_q*24 + mat_index_r]; // doesnt really matter in what order these indices are used, since the scoring table is symmetrical

      short diag_score = final_prev_prev_H + add_score;

      _curr_H = findMaxFour(diag_score, _curr_F, _curr_E, 0);

      int _i = inverse ? maxSize - i : i;
      int _j = inverse ? minSize - thread_Id - 1: thread_Id + 1;

      thread_max_i = (thread_max >= _curr_H) ? thread_max_i : _i;
      thread_max_j = (thread_max >= _curr_H) ? thread_max_j : _j;
      thread_max   = (thread_max >= _curr_H) ? thread_max : _curr_H;
      i++;
    } else {
      // we need these dummy shuffle operations for NVIDIA GPUs
      short valeShfl = sg.shuffle(_prev_E, laneId);
      short valheShfl =  sg.shuffle(_prev_H, laneId);
      short testShufll = sg.shuffle(_prev_prev_H, laneId);
    }

    group_barrier(gp);
  }

  thread_max = blockShuffleReduce_with_index(thread_max, thread_max_i, thread_max_j,
      minSize, locTots, locInds, locInds2, inverse, item);  // thread 0 will have the correct values

  if(inverse) {
    if(thread_Id == 0){
      if(lengthSeqA < lengthSeqB){
        seqB_align_begin[block_Id] = (thread_max_i);
        seqA_align_begin[block_Id] = (thread_max_j);
      }
      else{
        seqA_align_begin[block_Id] = (thread_max_i);
        seqB_align_begin[block_Id] = (thread_max_j);
      }
    }
  } else {  
    if(thread_Id == 0) {
       if(lengthSeqA < lengthSeqB) {
         seqB_align_end[block_Id] = thread_max_i;
         seqA_align_end[block_Id] = thread_max_j;
         top_scores[block_Id] = thread_max;
       }
       else {
         seqA_align_end[block_Id] = thread_max_i;
         seqB_align_end[block_Id] = thread_max_j;
         top_scores[block_Id] = thread_max;
       }
    }
  }
}
