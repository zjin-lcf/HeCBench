#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "utils.h"
#include "kernels.dp.cpp"

int main(int argc, char **argv) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  Option option;
  checkOption(argc, argv, option);
  std::vector<Read> reads;
  readFile(reads, option);

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

  int *d_lengths;
  d_lengths = sycl::malloc_device<int>(readsCount, q_ct1);
  q_ct1.memcpy(d_lengths, h_lengths, readsCount * sizeof(int)).wait();

  long *d_offsets;
  d_offsets = sycl::malloc_device<long>((1 + readsCount), q_ct1);
  q_ct1.memcpy(d_offsets, h_offsets, (1 + readsCount) * sizeof(long)).wait();

  char *d_reads;
  d_reads = sycl::malloc_device<char>(total_length, q_ct1);
  q_ct1.memcpy(d_reads, h_reads, total_length * sizeof(char)).wait();

  sycl::range<3> baseToNum_grid(1, 1, 128);
  sycl::range<3> baseToNum_block(1, 1, 128);

  /*
  DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(baseToNum_grid * baseToNum_block, baseToNum_block),
          [=](sycl::nd_item<3> item_ct1) {
             kernel_baseToNumber(d_reads, total_length, item_ct1);
          });
   });

  unsigned int *d_compressed;
  d_compressed = (unsigned int *)sycl::malloc_device(
      (total_length / 16) * sizeof(int), q_ct1);

  int *d_gaps;
  d_gaps = sycl::malloc_device<int>(readsCount, q_ct1);

  sycl::range<3> compress_grid(1, 1, (readsCount + 127) / 128);
  sycl::range<3> compress_block(1, 1, 128);
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<3>(compress_grid * compress_block, compress_block),
          [=](sycl::nd_item<3> item_ct1) {
             kernel_compressData(d_lengths, d_offsets, d_reads, d_compressed,
                                 d_gaps, readsCount, item_ct1);
          });
   });

  //createIndex(data, option);

  unsigned short* h_indexs = (unsigned short*) malloc (sizeof(unsigned short) * total_length);
  long* h_words = (long*) malloc (sizeof(long) * readsCount);

  unsigned short *d_indexs;
  d_indexs = sycl::malloc_device<unsigned short>(total_length, q_ct1);

  unsigned short *d_orders;
  d_orders = sycl::malloc_device<unsigned short>(total_length, q_ct1);

  long *d_words;
  d_words = sycl::malloc_device<long>(readsCount, q_ct1);

  int *d_magicBase;
  d_magicBase = sycl::malloc_device<int>((readsCount * 4), q_ct1);

  int wordLength = option.wordLength;

  sycl::range<3> index_grid(1, 1, (readsCount + 127) / 128);
  sycl::range<3> index_block(1, 1, 128);

  switch (wordLength) {
    case 4:
      /*
      DPCT1049:4: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_createIndex4(d_reads, d_lengths, d_offsets, d_indexs,
                                    d_orders, d_words, d_magicBase, readsCount,
                                    item_ct1);
             });
      });
      break;
    case 5:
      /*
      DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_createIndex5(d_reads, d_lengths, d_offsets, d_indexs,
                                    d_orders, d_words, d_magicBase, readsCount,
                                    item_ct1);
             });
      });
      break;
    case 6:
      /*
      DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_createIndex6(d_reads, d_lengths, d_offsets, d_indexs,
                                    d_orders, d_words, d_magicBase, readsCount,
                                    item_ct1);
             });
      });
      break;
    case 7:
      /*
      DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_createIndex7(d_reads, d_lengths, d_offsets, d_indexs,
                                    d_orders, d_words, d_magicBase, readsCount,
                                    item_ct1);
             });
      });
      break;
  }

  // createCutoff(data, option);
  float threshold = option.threshold;
  int *d_wordCutoff;
  d_wordCutoff = sycl::malloc_device<int>(readsCount, q_ct1);

  /*
  DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(index_grid * index_block, index_block),
                       [=](sycl::nd_item<3> item_ct1) {
                          kernel_createCutoff(threshold, wordLength, d_lengths,
                                              d_words, d_wordCutoff, readsCount,
                                              item_ct1);
                       });
   });

  // sd_ortIndex(data);
  q_ct1.memcpy(h_indexs, d_indexs, sizeof(unsigned short) * total_length)
      .wait();

  q_ct1.memcpy(h_offsets, d_offsets, sizeof(long) * (1 + readsCount)).wait();

  q_ct1.memcpy(h_words, d_words, sizeof(long) * readsCount).wait();

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);
  q_ct1.memcpy(d_indexs, h_indexs, sizeof(unsigned short) * total_length)
      .wait();

  /*
  DPCT1049:3: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(index_grid * index_block, index_block),
                       [=](sycl::nd_item<3> item_ct1) {
                          kernel_mergeIndex(d_offsets, d_indexs, d_orders,
                                            d_words, readsCount, item_ct1);
                       });
   });

  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  int *d_cluster;
  d_cluster = sycl::malloc_device<int>(readsCount, q_ct1);
  q_ct1.memcpy(d_cluster, h_cluster, sizeof(int) * readsCount).wait();

  unsigned short* table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(table, 0, 65536*sizeof(unsigned short));  // fill zero

  unsigned short *d_table;
  d_table = sycl::malloc_device<unsigned short>(65536, q_ct1);
  q_ct1.memcpy(d_table, table, 65536 * sizeof(unsigned short)).wait();

  int r = -1; // a shorthand for representative

  sycl::range<3> makeTable_grid(1, 1, 128);
  sycl::range<3> makeTable_block(1, 1, 128);
  sycl::range<3> cleanTable_grid(1, 1, 128);
  sycl::range<3> cleanTable_block(1, 1, 128);
  sycl::range<3> magic_grid(1, 1, (readsCount + 127) / 128);
  sycl::range<3> magic_block(1, 1, 128);
  sycl::range<3> filter_grid(1, 1, readsCount);
  sycl::range<3> filter_block(1, 1, 128);
  sycl::range<3> align_grid(1, 1, (readsCount + 127) / 128);
  sycl::range<3> align_block(1, 1, 128);

  while (r < readsCount) {  // clustering

    updateRepresentative(d_cluster, &r, readsCount);  // update representative
    if (r >= readsCount-1) {  // complete
      break;
    }
    //std::cout << r << "/" << readsCount << std::endl;

    /*
    DPCT1049:8: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<3>(makeTable_grid * makeTable_block,
                                            makeTable_block),
                          [=](sycl::nd_item<3> item_ct1) {
                             kernel_makeTable(d_offsets, d_indexs, d_orders,
                                              d_words, d_table, r, item_ct1);
                          });
      });

    /*
    DPCT1049:9: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(magic_grid * magic_block, magic_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_magic(threshold, d_lengths, d_magicBase, d_cluster, r,
                             readsCount, item_ct1);
             });
      });

    /*
    DPCT1049:10: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      q_ct1.submit([&](sycl::handler &cgh) {
         sycl::accessor<int, 1, sycl::access::mode::read_write,
                        sycl::access::target::local>
             result_acc_ct1(sycl::range<1>(128), cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(filter_grid * filter_block, filter_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_filter(threshold, wordLength, d_lengths, d_offsets,
                              d_indexs, d_orders, d_words, d_wordCutoff,
                              d_cluster, d_table, readsCount, item_ct1,
                              result_acc_ct1.get_pointer());
             });
      });

    /*
    DPCT1049:11: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(
             sycl::nd_range<3>(align_grid * align_block, align_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_align(threshold, d_lengths, d_offsets, d_compressed,
                             d_gaps, r, d_cluster, readsCount, item_ct1);
             });
      });

    /*
    DPCT1049:12: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      q_ct1.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(sycl::nd_range<3>(cleanTable_grid * cleanTable_block,
                                            cleanTable_block),
                          [=](sycl::nd_item<3> item_ct1) {
                             kernel_cleanTable(d_offsets, d_indexs, d_orders,
                                               d_words, d_table, r, item_ct1);
                          });
      });
  }

  q_ct1.memcpy(h_cluster, d_cluster, sizeof(int) * readsCount).wait();

  std::ofstream file(option.outputFile.c_str());
  int sum = 0;
  for (int i = 0; i < reads.size(); i++) {
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

  sycl::free(d_lengths, q_ct1);
  sycl::free(d_offsets, q_ct1);
  sycl::free(d_reads, q_ct1);
  sycl::free(d_compressed, q_ct1);
  sycl::free(d_gaps, q_ct1);
  sycl::free(d_indexs, q_ct1);
  sycl::free(d_orders, q_ct1);
  sycl::free(d_words, q_ct1);
  sycl::free(d_magicBase, q_ct1);
  sycl::free(d_wordCutoff, q_ct1);
  sycl::free(d_cluster, q_ct1);
  sycl::free(d_table, q_ct1);

  return 0;
}
