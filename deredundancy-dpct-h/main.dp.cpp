#define DPCT_USM_LEVEL_NONE
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
  d_lengths = (int *)dpct::dpct_malloc(readsCount * sizeof(int));
  dpct::dpct_memcpy(d_lengths, h_lengths, readsCount * sizeof(int),
                    dpct::host_to_device);

  long *d_offsets;
  d_offsets = (long *)dpct::dpct_malloc((1 + readsCount) * sizeof(long));
  dpct::dpct_memcpy(d_offsets, h_offsets, (1 + readsCount) * sizeof(long),
                    dpct::host_to_device);

  char *d_reads;
  d_reads = (char *)dpct::dpct_malloc(total_length * sizeof(char));
  dpct::dpct_memcpy(d_reads, h_reads, total_length * sizeof(char),
                    dpct::host_to_device);

  sycl::range<3> baseToNum_grid(1, 1, 128);
  sycl::range<3> baseToNum_block(1, 1, 128);

  /*
  DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   {
      dpct::buffer_t d_reads_buf_ct0 = dpct::get_buffer(d_reads);
      q_ct1.submit([&](sycl::handler &cgh) {
         auto d_reads_acc_ct0 =
             d_reads_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

         cgh.parallel_for(sycl::nd_range<3>(baseToNum_grid * baseToNum_block,
                                            baseToNum_block),
                          [=](sycl::nd_item<3> item_ct1) {
                             kernel_baseToNumber((char *)(&d_reads_acc_ct0[0]),
                                                 total_length, item_ct1);
                          });
      });
   }

  unsigned int *d_compressed;
  d_compressed =
      (unsigned int *)dpct::dpct_malloc((total_length / 16) * sizeof(int));

  int *d_gaps;
  d_gaps = (int *)dpct::dpct_malloc(readsCount * sizeof(int));

  sycl::range<3> compress_grid(1, 1, (readsCount + 127) / 128);
  sycl::range<3> compress_block(1, 1, 128);
  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   {
      dpct::buffer_t d_lengths_buf_ct0 = dpct::get_buffer(d_lengths);
      dpct::buffer_t d_offsets_buf_ct1 = dpct::get_buffer(d_offsets);
      dpct::buffer_t d_reads_buf_ct2 = dpct::get_buffer(d_reads);
      dpct::buffer_t d_compressed_buf_ct3 = dpct::get_buffer(d_compressed);
      dpct::buffer_t d_gaps_buf_ct4 = dpct::get_buffer(d_gaps);
      q_ct1.submit([&](sycl::handler &cgh) {
         auto d_lengths_acc_ct0 =
             d_lengths_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
         auto d_offsets_acc_ct1 =
             d_offsets_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
         auto d_reads_acc_ct2 =
             d_reads_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
         auto d_compressed_acc_ct3 =
             d_compressed_buf_ct3.get_access<sycl::access::mode::read_write>(
                 cgh);
         auto d_gaps_acc_ct4 =
             d_gaps_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(compress_grid * compress_block, compress_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_compressData((const int *)(&d_lengths_acc_ct0[0]),
                                    (const long *)(&d_offsets_acc_ct1[0]),
                                    (const char *)(&d_reads_acc_ct2[0]),
                                    (unsigned int *)(&d_compressed_acc_ct3[0]),
                                    (int *)(&d_gaps_acc_ct4[0]), readsCount,
                                    item_ct1);
             });
      });
   }

  //createIndex(data, option);

  unsigned short* h_indexs = (unsigned short*) malloc (sizeof(unsigned short) * total_length);
  long* h_words = (long*) malloc (sizeof(long) * readsCount);

  unsigned short *d_indexs;
  d_indexs = (unsigned short *)dpct::dpct_malloc(total_length *
                                                 sizeof(unsigned short));

  unsigned short *d_orders;
  d_orders = (unsigned short *)dpct::dpct_malloc(total_length *
                                                 sizeof(unsigned short));

  long *d_words;
  d_words = (long *)dpct::dpct_malloc(readsCount * sizeof(long));

  int *d_magicBase;
  d_magicBase = (int *)dpct::dpct_malloc((readsCount * 4) * sizeof(int));

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
      {
         dpct::buffer_t d_reads_buf_ct0 = dpct::get_buffer(d_reads);
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct2 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct3 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct4 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct5 = dpct::get_buffer(d_words);
         dpct::buffer_t d_magicBase_buf_ct6 = dpct::get_buffer(d_magicBase);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_reads_acc_ct0 =
                d_reads_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct2 =
                d_offsets_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct3 =
                d_indexs_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct4 =
                d_orders_buf_ct4.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct5 =
                d_words_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
            auto d_magicBase_acc_ct6 =
                d_magicBase_buf_ct6.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(index_grid * index_block, index_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_createIndex4((const char *)(&d_reads_acc_ct0[0]),
                                       (const int *)(&d_lengths_acc_ct1[0]),
                                       (const long *)(&d_offsets_acc_ct2[0]),
                                       (unsigned short *)(&d_indexs_acc_ct3[0]),
                                       (unsigned short *)(&d_orders_acc_ct4[0]),
                                       (long *)(&d_words_acc_ct5[0]),
                                       (int *)(&d_magicBase_acc_ct6[0]),
                                       readsCount, item_ct1);
                });
         });
      }
      break;
    case 5:
      /*
      DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      {
         dpct::buffer_t d_reads_buf_ct0 = dpct::get_buffer(d_reads);
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct2 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct3 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct4 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct5 = dpct::get_buffer(d_words);
         dpct::buffer_t d_magicBase_buf_ct6 = dpct::get_buffer(d_magicBase);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_reads_acc_ct0 =
                d_reads_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct2 =
                d_offsets_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct3 =
                d_indexs_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct4 =
                d_orders_buf_ct4.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct5 =
                d_words_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
            auto d_magicBase_acc_ct6 =
                d_magicBase_buf_ct6.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(index_grid * index_block, index_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_createIndex5((const char *)(&d_reads_acc_ct0[0]),
                                       (const int *)(&d_lengths_acc_ct1[0]),
                                       (const long *)(&d_offsets_acc_ct2[0]),
                                       (unsigned short *)(&d_indexs_acc_ct3[0]),
                                       (unsigned short *)(&d_orders_acc_ct4[0]),
                                       (long *)(&d_words_acc_ct5[0]),
                                       (int *)(&d_magicBase_acc_ct6[0]),
                                       readsCount, item_ct1);
                });
         });
      }
      break;
    case 6:
      /*
      DPCT1049:6: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      {
         dpct::buffer_t d_reads_buf_ct0 = dpct::get_buffer(d_reads);
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct2 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct3 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct4 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct5 = dpct::get_buffer(d_words);
         dpct::buffer_t d_magicBase_buf_ct6 = dpct::get_buffer(d_magicBase);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_reads_acc_ct0 =
                d_reads_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct2 =
                d_offsets_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct3 =
                d_indexs_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct4 =
                d_orders_buf_ct4.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct5 =
                d_words_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
            auto d_magicBase_acc_ct6 =
                d_magicBase_buf_ct6.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(index_grid * index_block, index_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_createIndex6((const char *)(&d_reads_acc_ct0[0]),
                                       (const int *)(&d_lengths_acc_ct1[0]),
                                       (const long *)(&d_offsets_acc_ct2[0]),
                                       (unsigned short *)(&d_indexs_acc_ct3[0]),
                                       (unsigned short *)(&d_orders_acc_ct4[0]),
                                       (long *)(&d_words_acc_ct5[0]),
                                       (int *)(&d_magicBase_acc_ct6[0]),
                                       readsCount, item_ct1);
                });
         });
      }
      break;
    case 7:
      /*
      DPCT1049:7: The workgroup size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the workgroup size if needed.
      */
      {
         dpct::buffer_t d_reads_buf_ct0 = dpct::get_buffer(d_reads);
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct2 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct3 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct4 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct5 = dpct::get_buffer(d_words);
         dpct::buffer_t d_magicBase_buf_ct6 = dpct::get_buffer(d_magicBase);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_reads_acc_ct0 =
                d_reads_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct2 =
                d_offsets_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct3 =
                d_indexs_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct4 =
                d_orders_buf_ct4.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct5 =
                d_words_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
            auto d_magicBase_acc_ct6 =
                d_magicBase_buf_ct6.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(index_grid * index_block, index_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_createIndex7((const char *)(&d_reads_acc_ct0[0]),
                                       (const int *)(&d_lengths_acc_ct1[0]),
                                       (const long *)(&d_offsets_acc_ct2[0]),
                                       (unsigned short *)(&d_indexs_acc_ct3[0]),
                                       (unsigned short *)(&d_orders_acc_ct4[0]),
                                       (long *)(&d_words_acc_ct5[0]),
                                       (int *)(&d_magicBase_acc_ct6[0]),
                                       readsCount, item_ct1);
                });
         });
      }
      break;
  }

  // createCutoff(data, option);
  float threshold = option.threshold;
  int *d_wordCutoff;
  d_wordCutoff = (int *)dpct::dpct_malloc(sizeof(int) * readsCount);

  /*
  DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   {
      dpct::buffer_t d_lengths_buf_ct2 = dpct::get_buffer(d_lengths);
      dpct::buffer_t d_words_buf_ct3 = dpct::get_buffer(d_words);
      dpct::buffer_t d_wordCutoff_buf_ct4 = dpct::get_buffer(d_wordCutoff);
      q_ct1.submit([&](sycl::handler &cgh) {
         auto d_lengths_acc_ct2 =
             d_lengths_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
         auto d_words_acc_ct3 =
             d_words_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
         auto d_wordCutoff_acc_ct4 =
             d_wordCutoff_buf_ct4.get_access<sycl::access::mode::read_write>(
                 cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_createCutoff(
                    threshold, wordLength, (const int *)(&d_lengths_acc_ct2[0]),
                    (long *)(&d_words_acc_ct3[0]),
                    (int *)(&d_wordCutoff_acc_ct4[0]), readsCount, item_ct1);
             });
      });
   }

  // sd_ortIndex(data);
  dpct::dpct_memcpy(h_indexs, d_indexs, sizeof(unsigned short) * total_length,
                    dpct::device_to_host);

  dpct::dpct_memcpy(h_offsets, d_offsets, sizeof(long) * (1 + readsCount),
                    dpct::device_to_host);

  dpct::dpct_memcpy(h_words, d_words, sizeof(long) * readsCount,
                    dpct::device_to_host);

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);
  dpct::dpct_memcpy(d_indexs, h_indexs, sizeof(unsigned short) * total_length,
                    dpct::host_to_device);

  /*
  DPCT1049:3: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
   {
      dpct::buffer_t d_offsets_buf_ct0 = dpct::get_buffer(d_offsets);
      dpct::buffer_t d_indexs_buf_ct1 = dpct::get_buffer(d_indexs);
      dpct::buffer_t d_orders_buf_ct2 = dpct::get_buffer(d_orders);
      dpct::buffer_t d_words_buf_ct3 = dpct::get_buffer(d_words);
      q_ct1.submit([&](sycl::handler &cgh) {
         auto d_offsets_acc_ct0 =
             d_offsets_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
         auto d_indexs_acc_ct1 =
             d_indexs_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
         auto d_orders_acc_ct2 =
             d_orders_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
         auto d_words_acc_ct3 =
             d_words_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);

         cgh.parallel_for(
             sycl::nd_range<3>(index_grid * index_block, index_block),
             [=](sycl::nd_item<3> item_ct1) {
                kernel_mergeIndex(
                    (const long *)(&d_offsets_acc_ct0[0]),
                    (const unsigned short *)(&d_indexs_acc_ct1[0]),
                    (unsigned short *)(&d_orders_acc_ct2[0]),
                    (const long *)(&d_words_acc_ct3[0]), readsCount, item_ct1);
             });
      });
   }

  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  int *d_cluster;
  d_cluster = (int *)dpct::dpct_malloc(sizeof(int) * readsCount);
  dpct::dpct_memcpy(d_cluster, h_cluster, sizeof(int) * readsCount,
                    dpct::host_to_device);

  unsigned short* table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(table, 0, 65536*sizeof(unsigned short));  // fill zero

  unsigned short *d_table;
  d_table = (unsigned short *)dpct::dpct_malloc(65536 * sizeof(unsigned short));
  dpct::dpct_memcpy(d_table, table, 65536 * sizeof(unsigned short),
                    dpct::host_to_device);

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
      {
         dpct::buffer_t d_offsets_buf_ct0 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct1 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct2 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct3 = dpct::get_buffer(d_words);
         dpct::buffer_t d_table_buf_ct4 = dpct::get_buffer(d_table);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_offsets_acc_ct0 =
                d_offsets_buf_ct0.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct1 =
                d_indexs_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct2 =
                d_orders_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct3 =
                d_words_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
            auto d_table_acc_ct4 =
                d_table_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(makeTable_grid * makeTable_block,
                                  makeTable_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_makeTable(
                       (const long *)(&d_offsets_acc_ct0[0]),
                       (const unsigned short *)(&d_indexs_acc_ct1[0]),
                       (const unsigned short *)(&d_orders_acc_ct2[0]),
                       (const long *)(&d_words_acc_ct3[0]),
                       (unsigned short *)(&d_table_acc_ct4[0]), r, item_ct1);
                });
         });
      }

    /*
    DPCT1049:9: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      {
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_magicBase_buf_ct2 = dpct::get_buffer(d_magicBase);
         dpct::buffer_t d_cluster_buf_ct3 = dpct::get_buffer(d_cluster);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_magicBase_acc_ct2 =
                d_magicBase_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_cluster_acc_ct3 =
                d_cluster_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(magic_grid * magic_block, magic_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_magic(threshold, (const int *)(&d_lengths_acc_ct1[0]),
                                (const int *)(&d_magicBase_acc_ct2[0]),
                                (int *)(&d_cluster_acc_ct3[0]), r, readsCount,
                                item_ct1);
                });
         });
      }

    /*
    DPCT1049:10: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      {
         dpct::buffer_t d_lengths_buf_ct2 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct3 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct4 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct5 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct6 = dpct::get_buffer(d_words);
         dpct::buffer_t d_wordCutoff_buf_ct7 = dpct::get_buffer(d_wordCutoff);
         dpct::buffer_t d_cluster_buf_ct8 = dpct::get_buffer(d_cluster);
         dpct::buffer_t d_table_buf_ct9 = dpct::get_buffer(d_table);
         q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                result_acc_ct1(sycl::range<1>(128), cgh);
            auto d_lengths_acc_ct2 =
                d_lengths_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct3 =
                d_offsets_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct4 =
                d_indexs_buf_ct4.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct5 =
                d_orders_buf_ct5.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct6 =
                d_words_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
            auto d_wordCutoff_acc_ct7 =
                d_wordCutoff_buf_ct7.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_cluster_acc_ct8 =
                d_cluster_buf_ct8.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_table_acc_ct9 =
                d_table_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(filter_grid * filter_block, filter_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_filter(threshold, wordLength,
                                 (const int *)(&d_lengths_acc_ct2[0]),
                                 (const long *)(&d_offsets_acc_ct3[0]),
                                 (const unsigned short *)(&d_indexs_acc_ct4[0]),
                                 (const unsigned short *)(&d_orders_acc_ct5[0]),
                                 (const long *)(&d_words_acc_ct6[0]),
                                 (const int *)(&d_wordCutoff_acc_ct7[0]),
                                 (int *)(&d_cluster_acc_ct8[0]),
                                 (const unsigned short *)(&d_table_acc_ct9[0]),
                                 readsCount, item_ct1,
                                 result_acc_ct1.get_pointer());
                });
         });
      }

    /*
    DPCT1049:11: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      {
         dpct::buffer_t d_lengths_buf_ct1 = dpct::get_buffer(d_lengths);
         dpct::buffer_t d_offsets_buf_ct2 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_compressed_buf_ct3 = dpct::get_buffer(d_compressed);
         dpct::buffer_t d_gaps_buf_ct4 = dpct::get_buffer(d_gaps);
         dpct::buffer_t d_cluster_buf_ct6 = dpct::get_buffer(d_cluster);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_lengths_acc_ct1 =
                d_lengths_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_offsets_acc_ct2 =
                d_offsets_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_compressed_acc_ct3 =
                d_compressed_buf_ct3.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_gaps_acc_ct4 =
                d_gaps_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
            auto d_cluster_acc_ct6 =
                d_cluster_buf_ct6.get_access<sycl::access::mode::read_write>(
                    cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(align_grid * align_block, align_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_align(
                       threshold, (const int *)(&d_lengths_acc_ct1[0]),
                       (const long *)(&d_offsets_acc_ct2[0]),
                       (const unsigned int *)(&d_compressed_acc_ct3[0]),
                       (const int *)(&d_gaps_acc_ct4[0]), r,
                       (int *)(&d_cluster_acc_ct6[0]), readsCount, item_ct1);
                });
         });
      }

    /*
    DPCT1049:12: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
      {
         dpct::buffer_t d_offsets_buf_ct0 = dpct::get_buffer(d_offsets);
         dpct::buffer_t d_indexs_buf_ct1 = dpct::get_buffer(d_indexs);
         dpct::buffer_t d_orders_buf_ct2 = dpct::get_buffer(d_orders);
         dpct::buffer_t d_words_buf_ct3 = dpct::get_buffer(d_words);
         dpct::buffer_t d_table_buf_ct4 = dpct::get_buffer(d_table);
         q_ct1.submit([&](sycl::handler &cgh) {
            auto d_offsets_acc_ct0 =
                d_offsets_buf_ct0.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_indexs_acc_ct1 =
                d_indexs_buf_ct1.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_orders_acc_ct2 =
                d_orders_buf_ct2.get_access<sycl::access::mode::read_write>(
                    cgh);
            auto d_words_acc_ct3 =
                d_words_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
            auto d_table_acc_ct4 =
                d_table_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(cleanTable_grid * cleanTable_block,
                                  cleanTable_block),
                [=](sycl::nd_item<3> item_ct1) {
                   kernel_cleanTable(
                       (const long *)(&d_offsets_acc_ct0[0]),
                       (const unsigned short *)(&d_indexs_acc_ct1[0]),
                       (const unsigned short *)(&d_orders_acc_ct2[0]),
                       (const long *)(&d_words_acc_ct3[0]),
                       (unsigned short *)(&d_table_acc_ct4[0]), r, item_ct1);
                });
         });
      }
  }

  dpct::dpct_memcpy(h_cluster, d_cluster, sizeof(int) * readsCount,
                    dpct::device_to_host);

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

  dpct::dpct_free(d_lengths);
  dpct::dpct_free(d_offsets);
  dpct::dpct_free(d_reads);
  dpct::dpct_free(d_compressed);
  dpct::dpct_free(d_gaps);
  dpct::dpct_free(d_indexs);
  dpct::dpct_free(d_orders);
  dpct::dpct_free(d_words);
  dpct::dpct_free(d_magicBase);
  dpct::dpct_free(d_wordCutoff);
  dpct::dpct_free(d_cluster);
  dpct::dpct_free(d_table);

  return 0;
}
