#include <sycl.hpp>
#include "utils.hpp"
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

  int its = (totalAlignments>20000)?(ceil((float)totalAlignments/20000)):1;
  unsigned NBLOCKS    = totalAlignments;
  unsigned leftOvers    = NBLOCKS % its;
  unsigned stringsPerIt = NBLOCKS / its;
  unsigned maxAlignments = stringsPerIt + leftOvers;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  short *d_ref_start = sycl::malloc_device<short>(maxAlignments, q);
  short *d_ref_end = sycl::malloc_device<short>(maxAlignments, q);
  short *d_query_start = sycl::malloc_device<short>(maxAlignments, q);
  short *d_query_end = sycl::malloc_device<short>(maxAlignments, q);
  short *d_scores = sycl::malloc_device<short>(maxAlignments, q);
  int *d_offset_ref = sycl::malloc_device<int>(maxAlignments, q);
  int *d_offset_query = sycl::malloc_device<int>(maxAlignments, q);
  char *d_strA = sycl::malloc_device<char>(maxContigSize * maxAlignments, q);
  char *d_strB = sycl::malloc_device<char>(maxReadSize * maxAlignments, q);
  short *d_encoding_matrix = sycl::malloc_device<short>(ENCOD_MAT_SIZE, q);
  q.memcpy(d_encoding_matrix, h_encoding_matrix, sizeof(short) * ENCOD_MAT_SIZE);

  short *d_scoring_matrix = sycl::malloc_device<short>(SCORE_MAT_SIZE, q);
  q.memcpy(d_scoring_matrix , h_scoring_matrix, sizeof(short) * SCORE_MAT_SIZE);

  short* ref_begin    = h_ref_begin;
  short* ref_end      = h_ref_end;  
  short* query_begin  = h_query_begin;
  short* query_end    = h_query_end;
  short* top_scores   = h_top_scores;   

  std::cout<<"Number of loop iterations: " << its << std::endl;

  q.wait();
  auto start = NOW;

  for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
  {
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

    q.memcpy(d_offset_ref, h_offsetA, sequences_per_stream * sizeof(int));
    q.memcpy(d_offset_query, h_offsetB, sequences_per_stream * sizeof(int));
    q.memcpy(d_strA, h_strA, totalLengthA * sizeof(char));
    q.memcpy(d_strB, h_strB, totalLengthB * sizeof(char));

    unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
    unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;
    printf("Shared memory bytes: %lu\n", ShmemBytes);
    printf("sequences per stream (SYCL grid size): %d\n", sequences_per_stream);
    printf("minSize (SYCL work-group size): %d\n", minSize);

    sycl::range<1> gws_aa(sequences_per_stream*minSize);
    sycl::range<1> lws_aa(minSize);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor< char, 1> is_valid_array(sycl::range<1>(ShmemBytes), cgh);
      sycl::local_accessor<short, 1> sh_prev_E(sycl::range<1>(32), cgh);
      sycl::local_accessor<short, 1> sh_prev_H(sycl::range<1>(32), cgh);
      sycl::local_accessor<short, 1> sh_prev_prev_H(sycl::range<1>(32), cgh);
      sycl::local_accessor<short, 1> sh_spill_prev_E(sycl::range<1>(1024), cgh);
      sycl::local_accessor<short, 1> sh_spill_prev_H(sycl::range<1>(1024), cgh);
      sycl::local_accessor<short, 1> sh_spill_prev_prev_H(sycl::range<1>(1024), cgh);
      sycl::local_accessor<short, 1> sh_aa_encoding(sycl::range<1>(ENCOD_MAT_SIZE), cgh);
      sycl::local_accessor<short, 1> sh_aa_scoring(sycl::range<1>(SCORE_MAT_SIZE), cgh);
      sycl::local_accessor<short, 1> sh_locTots(sycl::range<1>(32), cgh);
      sycl::local_accessor<short, 1> sh_locInds(sycl::range<1>(32), cgh);
      sycl::local_accessor<short, 1> sh_locInds2(sycl::range<1>(32), cgh);

      cgh.parallel_for<class aa>(
        sycl::nd_range<1>(gws_aa, lws_aa), [=] (sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(32)]] {
        sequence_aa_kernel(
           d_strA,
           d_strB,
           d_offset_ref,
           d_offset_query,
           d_ref_start,
           d_ref_end,
           d_query_start,
           d_query_end,
           d_scores,
           openGap,
           extendGap,
           d_scoring_matrix,
           d_encoding_matrix,
           is_valid_array.get_pointer(),
           sh_prev_E.get_pointer(),
           sh_prev_H.get_pointer(),
           sh_prev_prev_H.get_pointer(),
           sh_spill_prev_E.get_pointer(),
           sh_spill_prev_H.get_pointer(),
           sh_spill_prev_prev_H.get_pointer(),
           sh_aa_encoding.get_pointer(),
           sh_aa_scoring.get_pointer(),
	   sh_locTots.get_pointer(),
	   sh_locInds.get_pointer(),
	   sh_locInds2.get_pointer(),
           false,
           item
        );
      });
    });

    // copy back end index so that we can find new min
    q.memcpy(ref_end, d_ref_end, sizeof(short) * sequences_per_stream);
    q.memcpy(query_end, d_query_end, sizeof(short) * sequences_per_stream);
    
    // find the new largest of smaller lengths
    int newMin = get_new_min_length(ref_end, query_end, blocksLaunched);

    sycl::range<1> gws_aa_r(sequences_per_stream*newMin);
    sycl::range<1> lws_aa_r(newMin);

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<char, 1> is_valid_array(ShmemBytes, cgh);
      sycl::local_accessor<short, 1> sh_prev_E(32, cgh);
      sycl::local_accessor<short, 1> sh_prev_H(32, cgh);
      sycl::local_accessor<short, 1> sh_prev_prev_H(32, cgh);
      sycl::local_accessor<short, 1> local_spill_prev_E(1024, cgh);
      sycl::local_accessor<short, 1> local_spill_prev_H(1024, cgh);
      sycl::local_accessor<short, 1> local_spill_prev_prev_H(1024, cgh);
      sycl::local_accessor<short, 1> sh_aa_encoding(ENCOD_MAT_SIZE, cgh);
      sycl::local_accessor<short, 1> sh_aa_scoring(SCORE_MAT_SIZE, cgh);
      sycl::local_accessor<short, 1> locTots(32, cgh);
      sycl::local_accessor<short, 1> locInds(32, cgh);
      sycl::local_accessor<short, 1> locInds2(32, cgh);
      cgh.parallel_for<class aa_r>(
        sycl::nd_range<1>(gws_aa_r, lws_aa_r), [=] (sycl::nd_item<1> item)
        [[intel::reqd_sub_group_size(32)]] {
        sequence_aa_kernel(
           d_strA,
           d_strB,
           d_offset_ref,
           d_offset_query,
           d_ref_start,
           d_ref_end,
           d_query_start,
           d_query_end,
           d_scores,
           openGap,
           extendGap,
           d_scoring_matrix,
           d_encoding_matrix,
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


    q.memcpy(ref_begin, d_ref_start, sizeof(short) * sequences_per_stream);
    q.memcpy(query_begin, d_query_start, sizeof(short) * sequences_per_stream);
    q.memcpy(top_scores, d_scores, sizeof(short) * sequences_per_stream);


    ref_begin += stringsPerIt;
    query_begin += stringsPerIt;
    ref_end += stringsPerIt;
    query_end += stringsPerIt;
    top_scores += stringsPerIt;

  }  // iterations end here

  q.wait();
  auto end = NOW;

  sycl::free(d_ref_start, q);
  sycl::free(d_ref_end, q);
  sycl::free(d_query_start, q);
  sycl::free(d_query_end, q);
  sycl::free(d_scores, q);
  sycl::free(d_offset_ref, q);
  sycl::free(d_offset_query, q);
  sycl::free(d_strA, q);
  sycl::free(d_strB, q);
  sycl::free(d_encoding_matrix, q);
  sycl::free(d_scoring_matrix, q);

  std::chrono::duration<double> diff = end - start;
  std::cout << "Total Alignments:" << totalAlignments << "\n" 
            << "Max Reference Size:" << maxContigSize << "\n"
            << "Max Query Size:"<< maxReadSize << "\n" 
            << "Total loop iteration time (seconds):"<< diff.count() << "\n";

  std::ofstream results_file(filename);
  if (results_file.is_open()) {

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
  } else {
    std::cerr << "Error opening the result file "
              << filename << std::endl;
  }

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
