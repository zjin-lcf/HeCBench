#include <chrono>
#include "utils.h"
#include "kernels.cpp"

int main(int argc, char **argv) {
  Option option;
  checkOption(argc, argv, option);
  std::vector<Read> reads;
  bool fail = readFile(reads, option);
  if (fail) return 1;

  int readsCount = reads.size();
  int* h_lengths = (int*) malloc (sizeof(int) * readsCount);
  long* h_offsets = (long*) malloc (sizeof(long) * (1 + readsCount));

  h_offsets[0] = 0;
  for (int i = 0; i < readsCount; i++) {  // copy data for lengths and offsets
    int length = reads[i].data.size();
    h_lengths[i] = length;
    h_offsets[i+1] = h_offsets[i] + length/16*16+16;
  }

  long total_length = h_offsets[readsCount];

  char* h_reads = (char*) malloc (sizeof(char) * total_length);
  for (int i = 0; i < readsCount; i++) {  // copy data for reads
    memcpy(&h_reads[h_offsets[i]], reads[i].data.c_str(), h_lengths[i]*sizeof(char));
  }

  auto t1 = std::chrono::high_resolution_clock::now();

  unsigned int *h_compressed = (unsigned int*) malloc ((total_length / 16) * sizeof(int));

  int *h_gaps = (int*) malloc(readsCount * sizeof(int));

  unsigned short* h_indexs = (unsigned short*) malloc (total_length * sizeof(unsigned short));
  long* h_words = (long*) malloc (sizeof(long) * readsCount);

  unsigned short *h_orders = (unsigned short*) malloc (total_length * sizeof(unsigned short)); 

  int *h_magicBase = (int*) malloc ((readsCount * 4) * sizeof(int));  

  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  unsigned short* h_table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(h_table, 0, 65536*sizeof(unsigned short));  // fill zero

  int *h_wordCutoff = (int*) malloc (readsCount * sizeof(int));

#pragma omp target data map(to: h_lengths[0:readsCount], \
                                h_offsets[0:1+readsCount], \
                                h_reads[0:total_length], \
                                h_table[0:65536]) \
                        map(alloc: h_compressed[0:total_length/16], \
                                   h_gaps[0:readsCount], \
                                   h_indexs[0:total_length], \
                                   h_words[0:readsCount], \
                                   h_magicBase[0:readsCount*4], \
                                   h_orders[0:total_length], \
                                   h_wordCutoff[0:readsCount]), \
                        map(tofrom: h_cluster[0:readsCount])
{
  kernel_baseToNumber(h_reads, total_length);

  kernel_compressData(
      h_lengths,
      h_offsets, 
      h_reads, 
      h_compressed, 
      h_gaps, 
      readsCount);

  //createIndex(data, option);

  int wordLength = option.wordLength;


  switch (wordLength) {
    case 4:
      kernel_createIndex4(
          h_reads, 
          h_lengths,
          h_offsets, 
          h_indexs,
          h_orders,
          h_words, 
          h_magicBase, 
          readsCount);
      break;
    case 5:
      kernel_createIndex5(
          h_reads, 
          h_lengths,
          h_offsets, 
          h_indexs,
          h_orders,
          h_words, 
          h_magicBase, 
          readsCount);
      break;
    case 6:
      kernel_createIndex6(
          h_reads, 
          h_lengths,
          h_offsets, 
          h_indexs,
          h_orders,
          h_words, 
          h_magicBase, 
          readsCount);
      break;
    case 7:
      kernel_createIndex7(
          h_reads, 
          h_lengths,
          h_offsets, 
          h_indexs,
          h_orders,
          h_words, 
          h_magicBase, 
          readsCount);
      break;
  }

  // createCutoff(data, option);
  float threshold = option.threshold;

  kernel_createCutoff(
      threshold, 
      wordLength, 
      h_lengths,
      h_words,
      h_wordCutoff,
      readsCount);

  // sortIndex(data);
  #pragma omp target update from (h_indexs[0:total_length])
  #pragma omp target update from (h_offsets[0:1+readsCount])
  #pragma omp target update from (h_words[0:readsCount])

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);
  #pragma omp target update to (h_indexs[0:total_length])

  kernel_mergeIndex(
      h_offsets, 
      h_indexs, 
      h_orders,
      h_words, 
      readsCount);

  int r = -1; // a shorthand for representative

  while (r < readsCount) {  // clustering

    updateRepresentative(h_cluster, &r, readsCount);  // update representative
    if (r >= readsCount-1) {  // complete
      break;
    }
    //std::cout << r << "/" << readsCount << std::endl;

    kernel_makeTable(
        h_offsets, 
        h_indexs,
        h_orders,
        h_words,
        h_table,
        r);

    kernel_magic(
        threshold,
        h_lengths,
        h_magicBase,
        h_cluster,
        r,
        readsCount);

    kernel_filter(
        threshold, 
        wordLength, 
        h_lengths,
        h_offsets, 
        h_indexs,
        h_orders,
        h_words,
        h_wordCutoff,
        h_cluster,
        h_table,
        readsCount);

    kernel_align(
        threshold, 
        h_lengths, 
        h_offsets,
        h_compressed, 
        h_gaps, 
        r,
        h_cluster, 
        readsCount);

    kernel_cleanTable(
        h_offsets, 
        h_indexs,
        h_orders,
        h_words,
        h_table,
        r);
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Device offload time %lf secs \n", total_time / 1.0e6);
} // #pragma

  std::ofstream file(option.outputFile.c_str());
  int sum = 0;
  for (int i = 0; i < readsCount; i++) {
    if (h_cluster[i] == i) {
      file << reads[i].name << std::endl;
      file << reads[i].data << std::endl;
      sum++;
    }
  }
  file.close();

  std::cout << "cluster count: " << sum << std::endl;
  free(h_lengths);
  free(h_offsets);
  free(h_reads);
  free(h_indexs);
  free(h_words);
  free(h_cluster);
  free(h_table);

  return 0;
}
