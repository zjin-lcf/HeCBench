#include "utils.hpp"
#include "common.h"
#include "kernel.cpp"

void kernel_driver_aa(std::string filename,
		      std::vector<std::string> &reads, 
                      std::vector<std::string> &contigs,
                      short h_scoring_matrix[], 
                      short openGap, 
                      short extendGap)
{
  
  unsigned maxContigSize = getMaxLength(contigs);
  unsigned maxReadSize = getMaxLength(reads);
  unsigned totalAlignments = contigs.size(); 

  //std::cout <<"max contig:"<<maxContigSize<<std::endl;
  //std::cout <<"max read:"<<maxReadSize<<std::endl;
  //std::cout <<"total aligns:"<<totalAlignments<<std::endl;

  short* h_ref_begin    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_ref_end      = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_begin  = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_end    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_top_scores   = (short*) malloc (sizeof(short) * totalAlignments);
  unsigned* h_offsetA = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  unsigned* h_offsetB = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  char* h_strA = (char*) malloc (maxContigSize * totalAlignments);
  char* h_strB = (char*) malloc (maxContigSize * totalAlignments);

  short h_encoding_matrix[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    23,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,20,4,3,6,
    13,7,8,9,0,11,10,12,2,0,14,5,
    1,15,16,0,19,17,22,18,21};

  float total_packing;

  auto start = NOW;
  float total_time_cpu = 0;

  // total number of iterations
  int its = (totalAlignments>20000)?(ceil((float)totalAlignments/20000)):1;
  unsigned NBLOCKS    = totalAlignments;
  unsigned leftOvers    = NBLOCKS % its;
  unsigned stringsPerIt = NBLOCKS / its;
  unsigned maxAlignments = stringsPerIt + leftOvers;

  { // SYCL scope

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<short, 1> d_ref_start(maxAlignments);
  buffer<short, 1> d_ref_end(maxAlignments);
  buffer<short, 1> d_query_start(maxAlignments);
  buffer<short, 1> d_query_end(maxAlignments);
  buffer<short, 1> d_scores(maxAlignments);
  buffer<int, 1> d_offset_ref(maxAlignments);
  buffer<int, 1> d_offset_query(maxAlignments);
  buffer<char, 1> d_strA (maxContigSize * maxAlignments);
  buffer<char, 1> d_strB (maxReadSize * maxAlignments);
  buffer<short, 1> d_encoding_matrix (h_encoding_matrix, ENCOD_MAT_SIZE);
  buffer<short, 1> d_scoring_matrix (h_scoring_matrix, SCORE_MAT_SIZE);

  total_packing = 0;

  short* ref_begin    = h_ref_begin;
  short* ref_end      = h_ref_end;  
  short* query_begin  = h_query_begin;
  short* query_end    = h_query_end;
  short* top_scores   = h_top_scores;   

  std::cout<<"Number of loop iterations: " << its << std::endl;

  for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
  {
    auto packing_start = NOW;
    int  blocksLaunched;
    std::vector<std::string>::const_iterator beginAVec;
    std::vector<std::string>::const_iterator endAVec;
    std::vector<std::string>::const_iterator beginBVec;
    std::vector<std::string>::const_iterator endBVec;

    beginAVec = contigs.begin() + perGPUIts * stringsPerIt;
    beginBVec = reads.begin() + (perGPUIts * stringsPerIt);

    if(perGPUIts == its - 1)
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt + leftOvers;
      endBVec = reads.begin() + ((perGPUIts + 1) * stringsPerIt) + leftOvers;
      blocksLaunched = stringsPerIt + leftOvers;
    }
    else
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt;
      endBVec = reads.begin() +  (perGPUIts + 1) * stringsPerIt;
      blocksLaunched = stringsPerIt;
    }

    std::vector<std::string> sequencesA(beginAVec, endAVec);
    std::vector<std::string> sequencesB(beginBVec, endBVec);
    unsigned running_sum = 0;
    int sequences_per_stream = blocksLaunched;

    auto start_cpu = NOW;

    for(unsigned int i = 0; i < sequencesA.size(); i++)
    {
      running_sum +=sequencesA[i].size();
      h_offsetA[i] = running_sum;//sequencesA[i].size();
      //std::cout << "h_offsetA: " << h_offsetA[i] << std::endl;
    }
    unsigned totalLengthA = h_offsetA[sequencesA.size() - 1];
    // std::cout << "totalLengthA: " << totalLengthA << std::endl;

    running_sum = 0;
    for(unsigned int i = 0; i < sequencesB.size(); i++)
    {
      running_sum +=sequencesB[i].size();
      h_offsetB[i] = running_sum; //sequencesB[i].size();
      //std::cout << "h_offsetB: " << h_offsetB[i] << std::endl;
    }
    unsigned totalLengthB = h_offsetB[sequencesB.size() - 1];
    // std::cout << "totalLengthB: " << totalLengthB << std::endl;

    auto end_cpu = NOW;
    std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;

    total_time_cpu += cpu_dur.count();
    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;

    for(unsigned int i = 0; i < sequencesA.size(); i++)
    {
      char* seqptrA = h_strA + offsetSumA;
      memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
      char* seqptrB = h_strB + offsetSumB;
      memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
      offsetSumA += sequencesA[i].size();
      offsetSumB += sequencesB[i].size();
    }

    auto packing_end = NOW;
    std::chrono::duration<double> packing_dur = packing_end - packing_start;

    total_packing += packing_dur.count();

    q.submit([&] (handler &cgh) {
      auto offset_ref = d_offset_ref.get_access<sycl_discard_write>(cgh, range<1>(sequences_per_stream));
      cgh.copy(h_offsetA, offset_ref);
    });

    q.submit([&] (handler &cgh) {
      auto offset_query = d_offset_query.get_access<sycl_discard_write>(cgh, range<1>(sequences_per_stream));
      cgh.copy(h_offsetB, offset_query);
    });

    q.submit([&] (handler &cgh) {
      auto str = d_strA.get_access<sycl_discard_write>(cgh, range<1>(totalLengthA));
      cgh.copy(h_strA, str);  
    });

    q.submit([&] (handler &cgh) {
      auto str = d_strB.get_access<sycl_discard_write>(cgh, range<1>(totalLengthB));
      cgh.copy(h_strB, str);  
    });

    unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
    unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;
    printf("Shared memory bytes: %lu\n", ShmemBytes);
    printf("sequences per stream (SYCL grid size): %d\n", sequences_per_stream);
    printf("minSize (SYCL work-group size): %d\n", minSize);

    range<1> gws_aa(sequences_per_stream*minSize);
    range<1> lws_aa(minSize);

    q.submit([&] (handler &cgh) {
      auto strA = d_strA.get_access<sycl_read>(cgh);
      auto strB = d_strB.get_access<sycl_read>(cgh);
      auto offset_ref = d_offset_ref.get_access<sycl_read>(cgh);
      auto offset_query = d_offset_query.get_access<sycl_read>(cgh);
      auto ref_start = d_ref_start.get_access<sycl_read>(cgh);
      auto ref_end = d_ref_end.get_access<sycl_discard_write>(cgh);
      auto query_start = d_query_start.get_access<sycl_read>(cgh);
      auto query_end = d_query_end.get_access<sycl_discard_write>(cgh);
      auto scores = d_scores.get_access<sycl_discard_write>(cgh);
      auto scoring_matrix = d_scoring_matrix.get_access<sycl_read>(cgh);
      auto encoding_matrix = d_encoding_matrix.get_access<sycl_read>(cgh);

      accessor<char, 1, sycl_read_write, access::target::local> is_valid_array(ShmemBytes, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_E(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_H(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_prev_H(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_E(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_H(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_prev_H(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_aa_encoding(ENCOD_MAT_SIZE, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_aa_scoring(SCORE_MAT_SIZE, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locTots(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locInds(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locInds2(32, cgh);

      cgh.parallel_for<class aa>(nd_range<1>(gws_aa, lws_aa), [=] (nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
        sequence_aa_kernel(
           strA.get_pointer(),
           strB.get_pointer(),
           offset_ref.get_pointer(),
           offset_query.get_pointer(),
           ref_start.get_pointer(),
           ref_end.get_pointer(),
           query_start.get_pointer(),
           query_end.get_pointer(),
           scores.get_pointer(),
           openGap,
           extendGap,
           scoring_matrix.get_pointer(),
           encoding_matrix.get_pointer(),
           is_valid_array.get_pointer(),
           sh_prev_E.get_pointer(),
           sh_prev_H.get_pointer(),
           sh_prev_prev_H.get_pointer(),
           local_spill_prev_E.get_pointer(),
           local_spill_prev_H.get_pointer(),
           local_spill_prev_prev_H.get_pointer(),
           sh_aa_encoding.get_pointer(),
           sh_aa_scoring.get_pointer(),
	   locTots.get_pointer(),
	   locInds.get_pointer(),
	   locInds2.get_pointer(),
           false,
           item
        );
      });
    });

    // copyin back end index so that we can find new min
    q.submit([&] (handler &cgh) {
      auto acc = d_ref_end.get_access<sycl_read>(cgh, range<1>(sequences_per_stream));
      cgh.copy(acc, ref_end);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_query_end.get_access<sycl_read>(cgh, range<1>(sequences_per_stream));
      cgh.copy(acc, query_end);
    });

    q.wait();

    auto sec_cpu_start = NOW;

    // find the new largest of smaller lengths
    int newMin = get_new_min_length(ref_end, query_end, blocksLaunched);
    auto sec_cpu_end = NOW;
    std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
    total_time_cpu += dur_sec_cpu.count();

    range<1> gws_aa_r(sequences_per_stream*newMin);
    range<1> lws_aa_r(newMin);

    q.submit([&] (handler &cgh) {
      auto strA = d_strA.get_access<sycl_read>(cgh);
      auto strB = d_strB.get_access<sycl_read>(cgh);
      auto offset_ref = d_offset_ref.get_access<sycl_read>(cgh);
      auto offset_query = d_offset_query.get_access<sycl_read>(cgh);
      auto ref_start = d_ref_start.get_access<sycl_discard_write>(cgh);
      auto ref_end = d_ref_end.get_access<sycl_read>(cgh);
      auto query_start = d_query_start.get_access<sycl_discard_write>(cgh);
      auto query_end = d_query_end.get_access<sycl_read>(cgh);
      auto scores = d_scores.get_access<sycl_read>(cgh);
      auto scoring_matrix = d_scoring_matrix.get_access<sycl_read>(cgh);
      auto encoding_matrix = d_encoding_matrix.get_access<sycl_read>(cgh);
      accessor<char, 1, sycl_read_write, access::target::local> is_valid_array(ShmemBytes, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_E(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_H(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_prev_prev_H(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_E(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_H(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> local_spill_prev_prev_H(1024, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_aa_encoding(ENCOD_MAT_SIZE, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> sh_aa_scoring(SCORE_MAT_SIZE, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locTots(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locInds(32, cgh);
      accessor<short, 1, sycl_read_write, access::target::local> locInds2(32, cgh);
      cgh.parallel_for<class aa_r>(nd_range<1>(gws_aa_r, lws_aa_r), [=] (nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
        sequence_aa_kernel(
           strA.get_pointer(),
           strB.get_pointer(),
           offset_ref.get_pointer(),
           offset_query.get_pointer(),
           ref_start.get_pointer(),
           ref_end.get_pointer(),
           query_start.get_pointer(),
           query_end.get_pointer(),
           scores.get_pointer(),
           openGap,
           extendGap,
           scoring_matrix.get_pointer(),
           encoding_matrix.get_pointer(),
           is_valid_array.get_pointer(),
           sh_prev_E.get_pointer(),
           sh_prev_H.get_pointer(),
           sh_prev_prev_H.get_pointer(),
           local_spill_prev_E.get_pointer(),
           local_spill_prev_H.get_pointer(),
           local_spill_prev_prev_H.get_pointer(),
           sh_aa_encoding.get_pointer(),
           sh_aa_scoring.get_pointer(),
	   locTots.get_pointer(),
	   locInds.get_pointer(),
	   locInds2.get_pointer(),
           true,
           item);
      });
    });


    q.submit([&] (handler &cgh) {
      auto acc = d_ref_start.get_access<sycl_read>(cgh, range<1>(sequences_per_stream));
      cgh.copy(acc, ref_begin);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_query_start.get_access<sycl_read>(cgh, range<1>(sequences_per_stream));
      cgh.copy(acc, query_begin);
    });

    q.submit([&] (handler &cgh) {
      auto scores = d_scores.get_access<sycl_read>(cgh, range<1>(sequences_per_stream));
      cgh.copy(scores, top_scores);
    });

    ref_begin += stringsPerIt;
    query_begin += stringsPerIt;
    ref_end += stringsPerIt;
    query_end += stringsPerIt;
    top_scores += stringsPerIt;

  }  // iterations end here

  q.wait();

 } // SYCL scope 

  auto end  = NOW;

  std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
  std::cout <<"packing time:"<<total_packing<<std::endl;

  std::chrono::duration<double> diff = end - start;
  std::cout << "Total Alignments:" << totalAlignments << "\n" 
            << "Max Reference Size:" << maxContigSize << "\n"
            << "Max Query Size:"<< maxReadSize << "\n" 
            << "Total Execution Time (seconds):"<< diff.count() << "\n";

  std::ofstream results_file(filename);

  for(unsigned int k = 0; k < reads.size(); k++){
    results_file << h_top_scores[k] <<"\t"
      << h_ref_begin[k] <<"\t"
      << h_ref_end[k] - 1 <<"\t"
      << h_query_begin[k] <<"\t"
      << h_query_end[k] - 1
      << std::endl;
  }
  results_file.flush();
  results_file.close();

  long long int total_cells = 0;
  for(unsigned int l = 0; l < reads.size(); l++){
    total_cells += reads.at(l).size()*contigs.at(l).size();
  }

  std::cout << "Total Cells:"<<total_cells<<std::endl;

  free(h_ref_begin);
  free(h_ref_end);
  free(h_query_begin);
  free(h_query_end);
  free(h_top_scores);
  free(h_offsetA);
  free(h_offsetB);
  free(h_strA);
  free(h_strB);
}// end of amino acids kernel
