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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto t1 = std::chrono::high_resolution_clock::now();

  int *d_lengths = sycl::malloc_device<int>(readsCount, q);
  q.memcpy(d_lengths, h_lengths, sizeof(int) * readsCount);

  long *d_offsets = sycl::malloc_device<long>(1 + readsCount, q);
  q.memcpy(d_offsets, h_offsets, sizeof(long) * (1 + readsCount));

  char *d_reads = sycl::malloc_device<char>(total_length, q);
  q.memcpy(d_reads, h_reads, sizeof(char) * total_length);

  // copyData(reads, data);
  sycl::range<1> baseToNum_gws(128*128);
  sycl::range<1> baseToNum_lws(128);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class baseToNum> (
      sycl::nd_range<1>(baseToNum_gws, baseToNum_lws), [=] (sycl::nd_item<1> item) {
      kernel_baseToNumber(d_reads, total_length, item);
    });
  });

  unsigned int *d_compressed = sycl::malloc_device<unsigned int>(total_length / 16, q);
  int *d_gaps = sycl::malloc_device<int>(readsCount, q);

  sycl::range<1> compress_gws((readsCount+127)/128*128);
  sycl::range<1> compress_lws(128);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class compressData> (
      sycl::nd_range<1>(compress_gws, compress_lws), [=] (sycl::nd_item<1> item) {
      kernel_compressData(
        d_lengths,
        d_offsets,
        d_reads,
        d_compressed,
        d_gaps,
        readsCount,
        item);
    });
  });

  //createIndex(data, option);

  unsigned short* h_indexs = (unsigned short*) malloc (sizeof(unsigned short) * total_length);
  long* h_words = (long*) malloc (sizeof(long) * readsCount);

  unsigned short *d_indexs = sycl::malloc_device<unsigned short>(total_length, q);
  unsigned short *d_orders = sycl::malloc_device<unsigned short>(total_length, q);
  long *d_words = sycl::malloc_device<long>(readsCount, q);
  int *d_magicBase = sycl::malloc_device<int>(readsCount * 4, q);

  int wordLength = option.wordLength;

  sycl::range<1> index_gws ((readsCount+127)/128*128);
  sycl::range<1> index_lws (128);
  switch (wordLength) {
    case 4:
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class index4> (
          sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
          kernel_createIndex4(
            d_reads,
            d_lengths,
            d_offsets,
            d_indexs,
            d_orders,
            d_words,
            d_magicBase,
            readsCount,
            item);
        });
      });
      break;
    case 5:
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class index5> (
          sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
          kernel_createIndex5(
            d_reads,
            d_lengths,
            d_offsets,
            d_indexs,
            d_orders,
            d_words,
            d_magicBase,
            readsCount,
            item);
        });
      });
      break;
    case 6:
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class index6> (
          sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
          kernel_createIndex6(
            d_reads,
            d_lengths,
            d_offsets,
            d_indexs,
            d_orders,
            d_words,
            d_magicBase,
            readsCount,
            item);
        });
      });
      break;
    case 7:
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class index7> (
          sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
          kernel_createIndex7(
            d_reads,
            d_lengths,
            d_offsets,
            d_indexs,
            d_orders,
            d_words,
            d_magicBase,
            readsCount,
            item);
        });
      });
      break;
  }

  // createCutoff(data, option);
  float threshold = option.threshold;
  int *d_wordCutoff = sycl::malloc_device<int>(readsCount, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class createCutoff> (
      sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
      kernel_createCutoff(
        threshold,
        wordLength,
        d_lengths,
        d_words,
        d_wordCutoff,
        readsCount,
        item);
    });
  });

  // sortIndex(data);
  q.memcpy(h_indexs, d_indexs, sizeof(unsigned short) * total_length);

  q.memcpy(h_offsets, d_offsets, sizeof(long) * (1+readsCount));

  q.memcpy(h_words, d_words, sizeof(long) * readsCount);

  q.wait();

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);
  q.memcpy(d_indexs, h_indexs, sizeof(unsigned short) * total_length);

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class mergeIndex> (
      sycl::nd_range<1>(index_gws, index_lws), [=] (sycl::nd_item<1> item) {
      kernel_mergeIndex(
        d_offsets,
        d_indexs,
        d_orders,
        d_words,
        readsCount,
        item);
    });
  });

  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  int *d_cluster = sycl::malloc_device<int>(readsCount, q);
  q.memcpy(d_cluster, h_cluster, sizeof(int) * readsCount);

  unsigned short* table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(table, 0, 65536*sizeof(unsigned short));  // fill zero

  unsigned short* d_table = sycl::malloc_device<unsigned short>(65536, q);
  q.memcpy(d_table, table, 65536*sizeof(unsigned short));

  int r = -1; // a shorthand for representative

  sycl::range<1> makeTable_gws(128*128);
  sycl::range<1> makeTable_lws(128);
  sycl::range<1> cleanTable_gws(128*128);
  sycl::range<1> cleanTable_lws(128);
  sycl::range<1> magic_gws((readsCount+127)/128*128);
  sycl::range<1> magic_lws(128);
  sycl::range<1> filter_gws(readsCount*128);
  sycl::range<1> filter_lws(128);
  sycl::range<1> align_gws((readsCount+127)/128*128);
  sycl::range<1> align_lws(128);

  while (r < readsCount) {  // clustering

    updateRepresentative(q, d_cluster, &r, readsCount);  // update representative
    if (r >= readsCount-1) {  // complete
      break;
    }
    //std::cout << r << "/" << readsCount << std::endl;

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class makeTable>(
        sycl::nd_range<1>(makeTable_gws, makeTable_lws), [=] (sycl::nd_item<1> item) {
        kernel_makeTable(
          d_offsets,
          d_indexs,
          d_orders,
          d_words,
          d_table,
          r,
          item);
      });
    }); // create table

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class magic>(
        sycl::nd_range<1>(magic_gws, magic_lws), [=] (sycl::nd_item<1> item) {
        kernel_magic(
          threshold,
          d_lengths,
          d_magicBase,
          d_cluster,
          r,
          readsCount,
          item);
      });
    }); // magic filter


    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> result (sycl::range<1>(128), cgh);
      cgh.parallel_for<class filter>(
        sycl::nd_range<1>(filter_gws, filter_lws), [=] (sycl::nd_item<1> item) {
        kernel_filter(
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
          readsCount,
          item,
          result.get_pointer());
      });
    }); // word filter

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class align>(
        sycl::nd_range<1>(align_gws, align_lws), [=] (sycl::nd_item<1> item) {
        kernel_align(
          threshold,
          d_lengths,
          d_offsets,
          d_compressed,
          d_gaps,
          r,
          d_cluster,
          readsCount,
          item);
      });
    }); // dynamic programming

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class cleanTable>(
        sycl::nd_range<1>(cleanTable_gws, cleanTable_lws), [=] (sycl::nd_item<1> item) {
        kernel_cleanTable(
          d_offsets,
          d_indexs,
          d_orders,
          d_words,
          d_table,
          r,
          item);
      }); // table fill zero
    });
  }

  q.memcpy(h_cluster, d_cluster, sizeof(int) * readsCount).wait();

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

  sycl::free(d_lengths, q);
  sycl::free(d_offsets, q);
  sycl::free(d_reads, q);
  sycl::free(d_compressed, q);
  sycl::free(d_gaps, q);
  sycl::free(d_indexs, q);
  sycl::free(d_orders, q);
  sycl::free(d_words, q);
  sycl::free(d_magicBase, q);
  sycl::free(d_wordCutoff, q);
  sycl::free(d_cluster, q);
  sycl::free(d_table, q);
  return 0;
}
