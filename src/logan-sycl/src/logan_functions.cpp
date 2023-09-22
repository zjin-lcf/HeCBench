//==================================================================
// Title:  x-drop seed-and-extend alignment algorithm
// Author: A. Zeni, G. Guidi
//==================================================================
#include <chrono>
#include <sycl/sycl.hpp>
#include "logan_functions.hpp"
#include "seed.hpp"

using namespace std;
using namespace chrono;

inline void warpReduce(volatile short *input, int myTId)
{
  input[myTId] = (input[myTId] > input[myTId + 32]) ? input[myTId] : input[myTId + 32];
  input[myTId] = (input[myTId] > input[myTId + 16]) ? input[myTId] : input[myTId + 16];
  input[myTId] = (input[myTId] > input[myTId + 8]) ? input[myTId] : input[myTId + 8];
  input[myTId] = (input[myTId] > input[myTId + 4]) ? input[myTId] : input[myTId + 4];
  input[myTId] = (input[myTId] > input[myTId + 2]) ? input[myTId] : input[myTId + 2];
  input[myTId] = (input[myTId] > input[myTId + 1]) ? input[myTId] : input[myTId + 1];
}

inline short reduce_max(short *input, int dim, int n_threads, sycl::nd_item<1> &item)
{
  unsigned int myTId = item.get_local_id(0);
  if(dim>32){
    for(int i = n_threads/2; i >32; i>>=1){
      if(myTId < i){
        input[myTId] = (input[myTId] > input[myTId + i]) ? input[myTId] : input[myTId + i];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
  }
  if(myTId<32)
    warpReduce(input, myTId);

  item.barrier(sycl::access::fence_space::local_space);
  return input[0];
}

inline void updateExtendedSeedL(
    SeedL &seed,
    ExtensionDirectionL direction, //as there are only 4 directions we may consider even smaller data types
    int cols,
    int rows,
    int lowerDiag,
    int upperDiag)
{
  if (direction == EXTEND_LEFTL)
  {
    int beginDiag = seed.beginDiagonal;
    // Set lower and upper diagonals.

    if (getLowerDiagonal(seed) > beginDiag + lowerDiag)
      setLowerDiagonal(seed, beginDiag + lowerDiag);
    if (getUpperDiagonal(seed) < beginDiag + upperDiag)
      setUpperDiagonal(seed, beginDiag + upperDiag);

    // Set new start position of seed.
    seed.beginPositionH -= rows;
    seed.beginPositionV -= cols;
  } else {  // direction == EXTEND_RIGHTL
    // Set new lower and upper diagonals.
    int endDiag = seed.endDiagonal;
    if (getUpperDiagonal(seed) < endDiag - lowerDiag)
      setUpperDiagonal(seed, (endDiag - lowerDiag));
    if (getLowerDiagonal(seed) > (endDiag - upperDiag))
      setLowerDiagonal(seed, endDiag - upperDiag);

    // Set new end position of seed.
    seed.endPositionH += rows;
    seed.endPositionV += cols;

  }
}

inline void computeAntidiag(
    const short *antiDiag1,
    const short *antiDiag2,
          short *antiDiag3,
    const char *querySeg,
    const char *databaseSeg,
    const int best,
    const int scoreDropOff,
    const int cols,
    const int rows,
    const int minCol,
    const int maxCol,
    const int antiDiagNo,
    const int offset1,
    const int offset2,
    const ExtensionDirectionL direction,
    int n_threads,
    sycl::nd_item<1> &item)
{
  int tid = item.get_local_id(0);

  for(int i = 0; i < maxCol; i+=n_threads){

    int col = tid + minCol + i;
    int queryPos, dbPos;

    queryPos = col - 1;
    dbPos = col + rows - antiDiagNo - 1;

    if(col < maxCol){

      int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;

      int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;

      tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);

      antiDiag3[tid+1+i] = (tmp < best - scoreDropOff) ? UNDEF : tmp;

    }
  }
}

inline void calcExtendedLowerDiag(
    int &lowerDiag,
    int const &minCol,
    int const &antiDiagNo)
{
  int minRow = antiDiagNo - minCol;
  if (minCol - minRow < lowerDiag)
    lowerDiag = minCol - minRow;
}

inline void calcExtendedUpperDiag(
    int &upperDiag,
    int const &maxCol,
    int const &antiDiagNo)
{
  int maxRow = antiDiagNo + 1 - maxCol;
  if (maxCol - 1 - maxRow > upperDiag)
    upperDiag = maxCol - 1 - maxRow;
}

inline void initAntiDiag3(
    short *antiDiag3,
    int &a3size,
    int const &offset,
    int const &maxCol,
    int const &antiDiagNo,
    int const &minScore,
    int const &gapCost,
    int const &undefined)
{
  a3size = maxCol + 1 - offset;

  antiDiag3[0] = undefined;
  antiDiag3[maxCol - offset] = undefined;

  if (antiDiagNo * gapCost > minScore)
  {
    if (offset == 0) // init first column
      antiDiag3[0] = antiDiagNo * gapCost;
    if (antiDiagNo - maxCol == 0) // init first row
      antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
  }
}

inline void initAntiDiags(
    short *antiDiag1,
    short *antiDiag2,
    short *antiDiag3,
    int &a2size,
    int &a3size,
    int const &dropOff,
    int const &gapCost,
    int const &undefined)
{
  a2size = 1;

  antiDiag2[0] = 0;

  a3size = 2;

  antiDiag3[0] = gapCost;
  antiDiag3[1] = gapCost;
}

void extendSeedLGappedXDropOneDirectionGlobal(
    SeedL *__restrict seed,
    const char *__restrict querySegArray,
    const char *__restrict databaseSegArray,
    const ExtensionDirectionL direction,
    const int scoreDropOff,
    int *__restrict res,
    const int *__restrict offsetQuery,
    const int *__restrict offsetTarget,
    const int offAntidiag,
    short *__restrict antidiag,
    const int n_threads,
    sycl::nd_item<1> &item,
    short *__restrict temp)
{
  int myId = item.get_group(0);
  int myTId = item.get_local_id(0);
  const char *querySeg;
  const char *databaseSeg;

  if(myId==0){
    querySeg = querySegArray;
    databaseSeg = databaseSegArray;
  }
  else{
    querySeg = querySegArray + offsetQuery[myId-1];
    databaseSeg = databaseSegArray + offsetTarget[myId-1];
  }

  short *antiDiag1 = &antidiag[myId*offAntidiag*3];
  short* antiDiag2 = &antiDiag1[offAntidiag];
  short* antiDiag3 = &antiDiag2[offAntidiag];

  SeedL mySeed(seed[myId]);
  //dimension of the antidiagonals
  int a1size = 0, a2size = 0, a3size = 0;
  int cols, rows;

  if(myId == 0){
    cols = offsetQuery[myId]+1;
    rows = offsetTarget[myId]+1;
  }
  else{
    cols = offsetQuery[myId]-offsetQuery[myId-1]+1;
    rows = offsetTarget[myId]-offsetTarget[myId-1]+1;
  }

  if (rows == 1 || cols == 1) return;

  int minCol = 1;
  int maxCol = 2;

  int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
  int offset2 = 0; //                                                       in antiDiag2
  int offset3 = 0; //                                                       in antiDiag3

  initAntiDiags(antiDiag1,antiDiag2, antiDiag3, a2size, a3size, scoreDropOff, GAP_EXT, UNDEF);
  int antiDiagNo = 1; // the currently calculated anti-diagonal

  int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

  int lowerDiag = 0;
  int upperDiag = 0;

  while (minCol < maxCol)
  {
    ++antiDiagNo;

    //antidiagswap
    //antiDiag2 -> antiDiag1
    //antiDiag3 -> antiDiag2
    //antiDiag1 -> antiDiag3
    short *t = antiDiag1;
    antiDiag1 = antiDiag2;
    antiDiag2 = antiDiag3;
    antiDiag3 = t;
    int t_l = a1size;
    a1size = a2size;
    a2size = a3size;
    a3size = t_l;
    offset1 = offset2;
    offset2 = offset3;
    offset3 = minCol-1;

    initAntiDiag3(antiDiag3, a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, UNDEF);

    computeAntidiag(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg,
                    best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo,
                    offset1, offset2, direction, n_threads, item);
    //roofline analysis
    item.barrier(sycl::access::fence_space::local_space);

    int tmp, antiDiagBest = UNDEF;
    for(int i=0; i<a3size; i+=n_threads){
      int size = a3size-i;

      if(myTId<n_threads){
        temp[myTId] = (myTId<size) ? antiDiag3[myTId+i]:UNDEF;
      }

      item.barrier(sycl::access::fence_space::local_space);

      tmp = reduce_max(temp, size, n_threads, item);
      antiDiagBest = (tmp>antiDiagBest) ? tmp:antiDiagBest;

    }
    best = (best > antiDiagBest) ? best : antiDiagBest;

    while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == UNDEF &&
        minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == UNDEF)
    {
      ++minCol;
    }

    // Calculate new maxCol
    while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == UNDEF) &&
        (antiDiag2[maxCol - offset2 - 1] == UNDEF))
    {
      --maxCol;
    }
    ++maxCol;

    // Calculate new lowerDiag and upperDiag of extended seed
    calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
    calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

    // end of databaseSeg reached?
    minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
    // end of querySeg reached?
    maxCol = (maxCol < cols) ? maxCol : cols;
  }

  int longestExtensionCol = a3size + offset3 - 2;
  int longestExtensionRow = antiDiagNo - longestExtensionCol;
  int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];

  if (longestExtensionScore == UNDEF)
  {
    if (antiDiag2[a2size -2] != UNDEF)
    {
      // reached end of query segment
      longestExtensionCol = a2size + offset2 - 2;
      longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
      longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
    }
    else if (a2size > 2 && antiDiag2[a2size-3] != UNDEF)
    {
      // reached end of database segment
      longestExtensionCol = a2size + offset2 - 3;
      longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
      longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
    }
  }

  if (longestExtensionScore == UNDEF){

    // general case
    for (int i = 0; i < a1size; ++i){

      if (antiDiag1[i] > longestExtensionScore){

        longestExtensionScore = antiDiag1[i];
        longestExtensionCol = i + offset1;
        longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;

      }
    }
  }

  if (longestExtensionScore != UNDEF)
    updateExtendedSeedL(mySeed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);

  seed[myId] = mySeed;
  res[myId] = longestExtensionScore;
}

void extendSeedL(std::vector<SeedL> &seeds,
    ExtensionDirectionL direction,
    std::vector<std::string> &target,
    std::vector<std::string> &query,
    std::vector<ScoringSchemeL> &penalties,
    int const& XDrop,
    int const& kmer_length,
    int *res,
    int numAlignments,
    int ngpus,
    int n_threads
)
{

  if(scoreGapExtend(penalties[0]) >= 0){

    cout<<"Error: Logan does not support gap extension penalty >= 0\n";
    exit(-1);
  }
  if(scoreGapOpen(penalties[0]) >= 0){

    cout<<"Error: Logan does not support gap opening penalty >= 0\n";
    exit(-1);
  }

  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int deviceCount = gpu_devices.size();

  if (deviceCount == 0) {
    std::cout << "Error: no device found\n";
    return;
  }

  if (ngpus > deviceCount || ngpus > MAX_GPUS) {
    std::cout << "Error: the maximum number of devices allowed is "
              << std::min(deviceCount, MAX_GPUS) << std::endl;
    return;
  }


  //start measuring time
#ifdef ADAPTABLE
  n_threads = (XDrop/WARP_DIM + 1)* WARP_DIM;
  if(n_threads>1024)
    n_threads=1024;
#endif

  //declare streams
  sycl::queue stream_r[MAX_GPUS], stream_l[MAX_GPUS];

  for(int i = 0; i < ngpus; i++) {
    const auto& d = gpu_devices[i];
    stream_r[i] = sycl::queue(d, sycl::property::queue::in_order());
    stream_l[i] = sycl::queue(d, sycl::property::queue::in_order());
  }

  // NB nSequences is correlated to the number of GPUs that we have
  int nSequences = numAlignments/ngpus;
  int nSequencesLast = nSequences+numAlignments%ngpus;

  //final result of the alignment
  int *scoreLeft = (int *)malloc(numAlignments * sizeof(int));
  int *scoreRight = (int *)malloc(numAlignments * sizeof(int));

  //create two sets of seeds
  //copy seeds
  vector<SeedL> seeds_r;
  vector<SeedL> seeds_l;
  seeds_r.reserve(numAlignments);

  for (size_t i=0; i<seeds.size(); i++){
    seeds_r.push_back(seeds[i]);
  }

  //sequences offsets
  vector<int> offsetLeftQ[MAX_GPUS];
  vector<int> offsetLeftT[MAX_GPUS];
  vector<int> offsetRightQ[MAX_GPUS];
  vector<int> offsetRightT[MAX_GPUS];

  //shared_mem_size per block per GPU
  int ant_len_left[MAX_GPUS];
  int ant_len_right[MAX_GPUS];

  //antidiag in case shared memory isn't enough
  short *ant_l[MAX_GPUS], *ant_r[MAX_GPUS];

  //total lenght of the sequences
  int totalLengthQPref[MAX_GPUS];
  int totalLengthTPref[MAX_GPUS];
  int totalLengthQSuff[MAX_GPUS];
  int totalLengthTSuff[MAX_GPUS];

  //declare and allocate sequences prefixes and suffixes
  char *prefQ[MAX_GPUS], *prefT[MAX_GPUS];
  char *suffQ[MAX_GPUS], *suffT[MAX_GPUS];

  //declare GPU offsets
  int *offsetLeftQ_d[MAX_GPUS], *offsetLeftT_d[MAX_GPUS];
  int *offsetRightQ_d[MAX_GPUS], *offsetRightT_d[MAX_GPUS];

  //declare GPU results
  int *scoreLeft_d[MAX_GPUS], *scoreRight_d[MAX_GPUS];

  //declare GPU seeds
  SeedL *seed_d_l[MAX_GPUS], *seed_d_r[MAX_GPUS];

  //declare prefixes and suffixes on the GPU
  char *prefQ_d[MAX_GPUS], *prefT_d[MAX_GPUS];
  char *suffQ_d[MAX_GPUS], *suffT_d[MAX_GPUS];

  std::vector<double> pergpustime(ngpus);

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++){
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;
    //compute offsets and shared memory per block
    int MYTHREAD = omp_get_thread_num();
    auto start_setup_ithread = NOW;
    ant_len_left[i]=0;
    ant_len_right[i]=0;
    for(int j = 0; j < dim; j++){

      offsetLeftQ[i].push_back(getBeginPositionV(seeds[j+i*nSequences]));
      offsetLeftT[i].push_back(getBeginPositionH(seeds[j+i*nSequences]));
      ant_len_left[i] = std::max(std::min(offsetLeftQ[i][j],offsetLeftT[i][j]), ant_len_left[i]);

      offsetRightQ[i].push_back(query[j+i*nSequences].size()-getEndPositionV(seeds[j+i*nSequences]));
      offsetRightT[i].push_back(target[j+i*nSequences].size()-getEndPositionH(seeds[j+i*nSequences]));
      ant_len_right[i] = std::max(std::min(offsetRightQ[i][j], offsetRightT[i][j]), ant_len_right[i]);
    }

    //compute antidiagonal offsets
    partial_sum(offsetLeftQ[i].begin(),offsetLeftQ[i].end(),offsetLeftQ[i].begin());
    partial_sum(offsetLeftT[i].begin(),offsetLeftT[i].end(),offsetLeftT[i].begin());
    partial_sum(offsetRightQ[i].begin(),offsetRightQ[i].end(),offsetRightQ[i].begin());
    partial_sum(offsetRightT[i].begin(),offsetRightT[i].end(),offsetRightT[i].begin());
    //set total length of the sequences
    totalLengthQPref[i] = offsetLeftQ[i][dim-1];
    totalLengthTPref[i] = offsetLeftT[i][dim-1];
    totalLengthQSuff[i] = offsetRightQ[i][dim-1];
    totalLengthTSuff[i] = offsetRightT[i][dim-1];
    //allocate sequences prefix and suffix on the CPU
    prefQ[i] = (char*)malloc(sizeof(char)*totalLengthQPref[i]);
    prefT[i] = (char*)malloc(sizeof(char)*totalLengthTPref[i]);
    suffQ[i] = (char*)malloc(sizeof(char)*totalLengthQSuff[i]);
    suffT[i] = (char*)malloc(sizeof(char)*totalLengthTSuff[i]);
    //generate prefix and suffix on the CPU
    reverse_copy(query[0+i*nSequences].c_str(),query[0+i*nSequences].c_str()+offsetLeftQ[i][0],prefQ[i]);

    memcpy(prefT[i], target[0+i*nSequences].c_str(), offsetLeftT[i][0]);
    memcpy(suffQ[i], query[0+i*nSequences].c_str()+getEndPositionV(seeds[0+i*nSequences]), offsetRightQ[i][0]);
    reverse_copy(target[0+i*nSequences].c_str()+getEndPositionH(seeds[0+i*nSequences]),target[0+i*nSequences].c_str()+getEndPositionH(seeds[0+i*nSequences])+offsetRightT[i][0],suffT[i]);

    for(int j = 1; j<dim; j++){
      char *seqptr = prefQ[i] + offsetLeftQ[i][j-1];
      reverse_copy(query[j+i*nSequences].c_str(),query[j+i*nSequences].c_str()+(offsetLeftQ[i][j]-offsetLeftQ[i][j-1]),seqptr);

      seqptr = prefT[i] + offsetLeftT[i][j-1];
      memcpy(seqptr, target[j+i*nSequences].c_str(), offsetLeftT[i][j]-offsetLeftT[i][j-1]);
      seqptr = suffQ[i] + offsetRightQ[i][j-1];
      memcpy(seqptr, query[j+i*nSequences].c_str()+getEndPositionV(seeds[j+i*nSequences]), offsetRightQ[i][j]-offsetRightQ[i][j-1]);
      seqptr = suffT[i] + offsetRightT[i][j-1];
      reverse_copy(target[j+i*nSequences].c_str()+getEndPositionH(seeds[j+i*nSequences]),target[j+i*nSequences].c_str()+getEndPositionH(seeds[j+i*nSequences])+(offsetRightT[i][j]-offsetRightT[i][j-1]),seqptr);

    }
    auto end_setup_ithread = NOW;
    duration<double> setup_ithread = end_setup_ithread - start_setup_ithread;
    pergpustime[MYTHREAD] = setup_ithread.count();
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;

    //allocate antidiagonals on the GPU
    ant_l[i] = sycl::malloc_device<short>(ant_len_left[i]*3*dim, stream_l[i]);
    ant_r[i] = sycl::malloc_device<short>(ant_len_right[i]*3*dim, stream_r[i]);

    //allocate offsets on the GPU
    offsetLeftQ_d[i] = sycl::malloc_device<int>(dim, stream_l[i]);
    offsetLeftT_d[i] = sycl::malloc_device<int>(dim, stream_l[i]);
    offsetRightQ_d[i] = sycl::malloc_device<int>(dim, stream_r[i]);
    offsetRightT_d[i] = sycl::malloc_device<int>(dim, stream_r[i]);
    //allocate results on the GPU
    scoreLeft_d[i] = sycl::malloc_device<int>(dim, stream_l[i]);
    scoreRight_d[i] = sycl::malloc_device<int>(dim, stream_r[i]);
    //allocate seeds on the GPU
    seed_d_l[i] = sycl::malloc_device<SeedL>(dim, stream_l[i]);
    seed_d_r[i] = sycl::malloc_device<SeedL>(dim, stream_r[i]);
    //allocate sequences on the GPU
    prefQ_d[i] = sycl::malloc_device<char>(totalLengthQPref[i], stream_l[i]);
    prefT_d[i] = sycl::malloc_device<char>(totalLengthTPref[i], stream_l[i]);
    suffQ_d[i] = sycl::malloc_device<char>(totalLengthQSuff[i], stream_r[i]);
    suffT_d[i] = sycl::malloc_device<char>(totalLengthTSuff[i], stream_r[i]);
    //copy seeds to the GPU
    stream_l[i].memcpy(seed_d_l[i], &seeds[0] + i * nSequences, dim * sizeof(SeedL));
    stream_r[i].memcpy(seed_d_r[i], &seeds_r[0] + i * nSequences, dim * sizeof(SeedL));
    //copy offsets to the GPU
    stream_l[i].memcpy(offsetLeftQ_d[i], &offsetLeftQ[i][0], dim * sizeof(int));
    stream_l[i].memcpy(offsetLeftT_d[i], &offsetLeftT[i][0], dim * sizeof(int));
    stream_r[i].memcpy(offsetRightQ_d[i], &offsetRightQ[i][0], dim * sizeof(int));
    stream_r[i].memcpy(offsetRightT_d[i], &offsetRightT[i][0], dim * sizeof(int));
    //copy sequences to the GPU
    stream_l[i].memcpy(prefQ_d[i], prefQ[i], totalLengthQPref[i] * sizeof(char));
    stream_l[i].memcpy(prefT_d[i], prefT[i], totalLengthTPref[i] * sizeof(char));
    stream_r[i].memcpy(suffQ_d[i], suffQ[i], totalLengthQSuff[i] * sizeof(char));
    stream_r[i].memcpy(suffT_d[i], suffT[i], totalLengthTSuff[i] * sizeof(char));
  }

  auto start_c = NOW;

  //execute kernels
  #pragma omp parallel for
  for(int i = 0; i<ngpus;i++)
  {
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;
    sycl::range<1> gws (dim * n_threads);
    sycl::range<1> lws (n_threads);

    stream_l[i].submit([&](sycl::handler &cgh) {
      sycl::local_accessor<short> sm (sycl::range<1>(n_threads), cgh);

      auto seed_d_l_i = seed_d_l[i];
      auto prefQ_d_i = prefQ_d[i];
      auto prefT_d_i = prefT_d[i];
      auto EXTEND_LEFTL_i = EXTEND_LEFTL;
      auto scoreLeft_d_i = scoreLeft_d[i];
      auto offsetLeftQ_d_i = offsetLeftQ_d[i];
      auto offsetLeftT_d_i = offsetLeftT_d[i];
      auto ant_len_left_i = ant_len_left[i];
      auto ant_l_i = ant_l[i];

      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        extendSeedLGappedXDropOneDirectionGlobal(
          seed_d_l_i, prefQ_d_i, prefT_d_i,
          EXTEND_LEFTL_i, XDrop, scoreLeft_d_i,
          offsetLeftQ_d_i, offsetLeftT_d_i,
          ant_len_left_i, ant_l_i, n_threads,
          item, sm.get_pointer());
      });
    });

    stream_r[i].submit([&](sycl::handler &cgh) {
      sycl::local_accessor<short> sm (sycl::range<1>(n_threads), cgh);

      auto seed_d_r_i = seed_d_r[i];
      auto suffQ_d_i = suffQ_d[i];
      auto suffT_d_i = suffT_d[i];
      auto EXTEND_RIGHTL_i = EXTEND_RIGHTL;
      auto scoreRight_d_i = scoreRight_d[i];
      auto offsetRightQ_d_i = offsetRightQ_d[i];
      auto offsetRightT_d_i = offsetRightT_d[i];
      auto ant_len_right_i = ant_len_right[i];
      auto ant_r_i = ant_r[i];

      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        extendSeedLGappedXDropOneDirectionGlobal(
          seed_d_r_i, suffQ_d_i, suffT_d_i,
          EXTEND_RIGHTL_i, XDrop, scoreRight_d_i,
          offsetRightQ_d_i, offsetRightT_d_i,
          ant_len_right_i, ant_r_i, n_threads,
          item, sm.get_pointer());
      });
    });
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;

    stream_l[i].memcpy(scoreLeft + i * nSequences, scoreLeft_d[i], dim * sizeof(int));
    stream_l[i].memcpy(&seeds[0] + i * nSequences, seed_d_l[i], dim * sizeof(SeedL));
    stream_r[i].memcpy(scoreRight + i * nSequences, scoreRight_d[i], dim * sizeof(int));
    stream_r[i].memcpy(&seeds_r[0] + i * nSequences, seed_d_r[i], dim * sizeof(SeedL));
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
    stream_l[i].wait();
    stream_r[i].wait();
  }

  auto end_c = NOW;
  duration<double> compute = end_c - start_c;
  std::cout << "Device only time [seconds]:\t" << compute.count() << std::endl;

#pragma omp parallel for
  for(int i = 0; i < ngpus; i++){
    free(prefQ[i]);
    free(prefT[i]);
    free(suffQ[i]);
    free(suffT[i]);
    sycl::free(prefQ_d[i], stream_l[i]);
    sycl::free(prefT_d[i], stream_l[i]);
    sycl::free(suffQ_d[i], stream_r[i]);
    sycl::free(suffT_d[i], stream_r[i]);
    sycl::free(offsetLeftQ_d[i], stream_l[i]);
    sycl::free(offsetLeftT_d[i], stream_l[i]);
    sycl::free(offsetRightQ_d[i], stream_r[i]);
    sycl::free(offsetRightT_d[i], stream_r[i]);
    sycl::free(seed_d_l[i], stream_l[i]);
    sycl::free(seed_d_r[i], stream_r[i]);
    sycl::free(scoreLeft_d[i], stream_l[i]);
    sycl::free(scoreRight_d[i], stream_r[i]);
    sycl::free(ant_l[i], stream_l[i]);
    sycl::free(ant_r[i], stream_r[i]);
  }

  for(int i = 0; i < numAlignments; i++){
    res[i] = scoreLeft[i]+scoreRight[i]+kmer_length;
    setEndPositionH(seeds[i], getEndPositionH(seeds_r[i]));
    setEndPositionV(seeds[i], getEndPositionV(seeds_r[i]));
    std::cout << res[i] << std::endl;
  }

  free(scoreLeft);
  free(scoreRight);
}
