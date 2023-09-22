#include <chrono>
#include "utils.h"
#include "kernels.cu"

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

  int *d_lengths; 
  hipMalloc((void**)&d_lengths, readsCount * sizeof(int));
  hipMemcpy(d_lengths, h_lengths, readsCount * sizeof(int), hipMemcpyHostToDevice);

  long *d_offsets; 
  hipMalloc((void**)&d_offsets, (1+readsCount) * sizeof(long));
  hipMemcpy(d_offsets, h_offsets, (1+readsCount) * sizeof(long), hipMemcpyHostToDevice);

  char *d_reads; 
  hipMalloc((void**)&d_reads, total_length * sizeof(char));
  hipMemcpy(d_reads, h_reads, total_length * sizeof(char), hipMemcpyHostToDevice);

  dim3 baseToNum_grid(128);
  dim3 baseToNum_block(128);

  hipLaunchKernelGGL(kernel_baseToNumber, dim3(baseToNum_grid), dim3(baseToNum_block), 0, 0, d_reads, total_length);

  unsigned int *d_compressed;
  hipMalloc((void**)&d_compressed, (total_length / 16) * sizeof(int));

  int *d_gaps;
  hipMalloc((void**)&d_gaps, readsCount * sizeof(int));

  dim3 compress_grid((readsCount+127)/128);
  dim3 compress_block(128);
  hipLaunchKernelGGL(kernel_compressData, dim3(compress_grid), dim3(compress_block), 0, 0, 
      d_lengths,
      d_offsets, 
      d_reads, 
      d_compressed, 
      d_gaps, 
      readsCount);

  //createIndex(data, option);

  unsigned short* h_indexs = (unsigned short*) malloc (sizeof(unsigned short) * total_length);
  long* h_words = (long*) malloc (sizeof(long) * readsCount);

  unsigned short *d_indexs;
  hipMalloc((void**)&d_indexs, total_length * sizeof(unsigned short)); 

  unsigned short *d_orders;
  hipMalloc((void**)&d_orders, total_length * sizeof(unsigned short)); 

  long *d_words;
  hipMalloc((void**)&d_words, readsCount * sizeof(long)); 

  int *d_magicBase;
  hipMalloc((void**)&d_magicBase, (readsCount * 4) * sizeof(int)); 

  int wordLength = option.wordLength;

  dim3 index_grid ((readsCount+127)/128);
  dim3 index_block (128);

  switch (wordLength) {
    case 4:
      hipLaunchKernelGGL(kernel_createIndex4, dim3(index_grid), dim3(index_block), 0, 0, 
          d_reads, 
          d_lengths,
          d_offsets, 
          d_indexs,
          d_orders,
          d_words, 
          d_magicBase, 
          readsCount);
      break;
    case 5:
      hipLaunchKernelGGL(kernel_createIndex5, dim3(index_grid), dim3(index_block), 0, 0, 
          d_reads, 
          d_lengths,
          d_offsets, 
          d_indexs,
          d_orders,
          d_words, 
          d_magicBase, 
          readsCount);
      break;
    case 6:
      hipLaunchKernelGGL(kernel_createIndex6, dim3(index_grid), dim3(index_block), 0, 0, 
          d_reads, 
          d_lengths,
          d_offsets, 
          d_indexs,
          d_orders,
          d_words, 
          d_magicBase, 
          readsCount);
      break;
    case 7:
      hipLaunchKernelGGL(kernel_createIndex7, dim3(index_grid), dim3(index_block), 0, 0, 
          d_reads, 
          d_lengths,
          d_offsets, 
          d_indexs,
          d_orders,
          d_words, 
          d_magicBase, 
          readsCount);
      break;
  }

  // createCutoff(data, option);
  float threshold = option.threshold;
  int *d_wordCutoff;
  hipMalloc((void**)&d_wordCutoff,  sizeof(int) * readsCount);

  hipLaunchKernelGGL(kernel_createCutoff, dim3(index_grid), dim3(index_block), 0, 0, 
      threshold, 
      wordLength, 
      d_lengths,
      d_words,
      d_wordCutoff,
      readsCount);

  // sd_ortIndex(data);
  hipMemcpy(h_indexs, d_indexs, sizeof(unsigned short) * total_length, hipMemcpyDeviceToHost); 

  hipMemcpy(h_offsets, d_offsets, sizeof(long) * (1+readsCount), hipMemcpyDeviceToHost); 

  hipMemcpy(h_words, d_words, sizeof(long) * readsCount, hipMemcpyDeviceToHost); 

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);
  hipMemcpy(d_indexs, h_indexs, sizeof(unsigned short) * total_length, hipMemcpyHostToDevice); 

  hipLaunchKernelGGL(kernel_mergeIndex, dim3(index_grid), dim3(index_block), 0, 0, 
      d_offsets, 
      d_indexs, 
      d_orders,
      d_words, 
      readsCount);

  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  int *d_cluster;
  hipMalloc((void**)&d_cluster, sizeof(int) * readsCount);
  hipMemcpy(d_cluster, h_cluster, sizeof(int) * readsCount, hipMemcpyHostToDevice);

  unsigned short* table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(table, 0, 65536*sizeof(unsigned short));  // fill zero

  unsigned short *d_table;
  hipMalloc((void**)&d_table, 65536*sizeof(unsigned short));
  hipMemcpy(d_table, table, 65536*sizeof(unsigned short), hipMemcpyHostToDevice);

  int r = -1; // a shorthand for representative

  dim3 makeTable_grid(128);
  dim3 makeTable_block(128);
  dim3 cleanTable_grid(128);
  dim3 cleanTable_block(128);
  dim3 magic_grid((readsCount+127)/128);
  dim3 magic_block(128);
  dim3 filter_grid(readsCount);
  dim3 filter_block(128);
  dim3 align_grid((readsCount+127)/128);
  dim3 align_block(128);

  while (r < readsCount) {  // clustering

    updateRepresentative(d_cluster, &r, readsCount);  // update representative
    if (r >= readsCount-1) {  // complete
      break;
    }
    //std::cout << r << "/" << readsCount << std::endl;

    hipLaunchKernelGGL(kernel_makeTable, dim3(makeTable_grid), dim3(makeTable_block), 0, 0, 
        d_offsets, 
        d_indexs,
        d_orders,
        d_words,
        d_table,
        r);

    hipLaunchKernelGGL(kernel_magic, dim3(magic_grid), dim3(magic_block), 0, 0, 
        threshold,
        d_lengths,
        d_magicBase,
        d_cluster,
        r,
        readsCount);

    hipLaunchKernelGGL(kernel_filter, dim3(filter_grid), dim3(filter_block), 0, 0, 
        threshold, 
        wordLength, 
        d_lengths,
        d_offsets, 
        d_indexs,
        d_orders,
        d_words,
        d_wordCutoff,
        d_cluster,
        d_table,
        readsCount);

    hipLaunchKernelGGL(kernel_align, dim3(align_grid), dim3(align_block), 0, 0, 
        threshold, 
        d_lengths, 
        d_offsets,
        d_compressed, 
        d_gaps, 
        r,
        d_cluster, 
        readsCount);

    hipLaunchKernelGGL(kernel_cleanTable, dim3(cleanTable_grid), dim3(cleanTable_block), 0, 0, 
        d_offsets, 
        d_indexs,
        d_orders,
        d_words,
        d_table,
        r);
  }

  hipMemcpy(h_cluster, d_cluster, sizeof(int) * readsCount, hipMemcpyDeviceToHost);

  auto t2 = std::chrono::high_resolution_clock::now();
  double total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  printf("Device offload time %lf secs \n", total_time / 1.0e6);

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
  free(table);

  hipFree(d_lengths);
  hipFree(d_offsets);
  hipFree(d_reads);
  hipFree(d_compressed);
  hipFree(d_gaps);
  hipFree(d_indexs);
  hipFree(d_orders);
  hipFree(d_words);
  hipFree(d_magicBase);
  hipFree(d_wordCutoff);
  hipFree(d_cluster);
  hipFree(d_table);

  return 0;
}
