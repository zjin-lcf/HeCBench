#include "utils.h"
#include "kernels.cpp"

int main(int argc, char **argv) {
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> d_lengths (h_lengths, readsCount);
  buffer<long, 1> d_offsets (h_offsets, (1 + readsCount));
  buffer<char, 1> d_reads (h_reads, total_length);
  d_reads.set_final_data(nullptr);

  // copyData(reads, data);
  range<1> baseToNum_gws(128*128);
  range<1> baseToNum_lws(128);

  q.submit([&](handler &cgh) {
    auto reads = d_reads.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class baseToNum> (nd_range<1>(baseToNum_gws, baseToNum_lws), [=] (nd_item<1> item) {
      kernel_baseToNumber(reads.get_pointer(), total_length, item);
    });
  });

  printf("baseToNum:\n");
  auto reads_acc = reads.get_access<sycl_read>();
  for (int i = 0; i < total_length; i++) printf("%d\n", reads_acc[i]);

  buffer<unsigned int> d_compressed(total_length / 16);
  buffer<int> d_gaps(readsCount);
  range<1> compress_gws((readsCount+127)/128*128);
  range<1> compress_lws(128);
  q.submit([&](handler &cgh) {
    auto lengths = d_lengths.get_access<sycl_read>(cgh);
    auto reads = d_reads.get_access<sycl_read>(cgh);
    auto offsets = d_offsets.get_access<sycl_read>(cgh);
    auto compressed = d_compressed.get_access<sycl_discard_write>(cgh);
    auto gaps = d_gaps.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class compressData> (nd_range<1>(compress_gws, compress_lws), [=] (nd_item<1> item) {
      kernel_compressData(
        lengths.get_pointer(),
        offsets.get_pointer(), 
        reads.get_pointer(), 
        compressed.get_pointer(), 
        gaps.get_pointer(), 
        readsCount, 
        item);
    });
  });

  printf("gaps:\n");
  auto gaps_acc = d_gaps.get_access<sycl_read>();
  for (int i = 0; i < readsCount; i++) printf("%d\n", gaps_acc[i]);

  printf("compressed:\n");
  auto compressed_acc = d_compressed.get_access<sycl_read>();
  for (int i = 0; i < total_length/16; i++) printf("%d\n", compressed_acc[i]);
  //createIndex(data, option);

  unsigned short* h_indexs = (unsigned short*) malloc (sizeof(unsigned short) * total_length);
  long* h_words = (long*) malloc (sizeof(long) * (readsCount + 1));

  buffer<unsigned short, 1> d_indexs (total_length);
  buffer<unsigned short, 1> d_orders (total_length);
  buffer<long, 1> d_words (readsCount + 1);
  buffer<int, 1> d_magicBase (readsCount * 4);

  int wordLength = option.wordLength;

  range<1> index_gws ((readsCount+127)/128*128);
  range<1> index_lws (128);
  switch (wordLength) {
    case 4:
      q.submit([&](handler &cgh) {
        auto reads = d_reads.get_access<sycl_read>(cgh);
        auto lengths = d_lengths.get_access<sycl_read>(cgh);
        auto offsets = d_offsets.get_access<sycl_read>(cgh);
        auto indexs = d_indexs.get_access<sycl_discard_write>(cgh);
        auto orders = d_orders.get_access<sycl_read>(cgh);
        auto words = d_words.get_access<sycl_discard_write>(cgh);
        auto magicBase = d_magicBase.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class index4> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
            kernel_createIndex4(
                reads.get_pointer(), 
                lengths.get_pointer(),
                offsets.get_pointer(), 
                indexs.get_pointer(),
                orders.get_pointer(),
                words.get_pointer(), 
                magicBase.get_pointer(), 
                readsCount,
                item);
        });
      });
      break;
    case 5:
      q.submit([&](handler &cgh) {
        auto reads = d_reads.get_access<sycl_read>(cgh);
        auto lengths = d_lengths.get_access<sycl_read>(cgh);
        auto offsets = d_offsets.get_access<sycl_read>(cgh);
        auto indexs = d_indexs.get_access<sycl_discard_write>(cgh);
        auto orders = d_orders.get_access<sycl_read>(cgh);
        auto words = d_words.get_access<sycl_discard_write>(cgh);
        auto magicBase = d_magicBase.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class index5> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
            kernel_createIndex5(
                reads.get_pointer(), 
                lengths.get_pointer(),
                offsets.get_pointer(), 
                indexs.get_pointer(),
                orders.get_pointer(),
                words.get_pointer(), 
                magicBase.get_pointer(), 
                readsCount,
                item);
        });
      });
      break;
    case 6:
      q.submit([&](handler &cgh) {
        auto reads = d_reads.get_access<sycl_read>(cgh);
        auto lengths = d_lengths.get_access<sycl_read>(cgh);
        auto offsets = d_offsets.get_access<sycl_read>(cgh);
        auto indexs = d_indexs.get_access<sycl_discard_write>(cgh);
        auto orders = d_orders.get_access<sycl_read>(cgh);
        auto words = d_words.get_access<sycl_discard_write>(cgh);
        auto magicBase = d_magicBase.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class index6> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
            kernel_createIndex6(
                reads.get_pointer(), 
                lengths.get_pointer(),
                offsets.get_pointer(), 
                indexs.get_pointer(),
                orders.get_pointer(),
                words.get_pointer(), 
                magicBase.get_pointer(), 
                readsCount,
                item);
        });
      });
      break;
    case 7:
      q.submit([&](handler &cgh) {
        auto reads = d_reads.get_access<sycl_read>(cgh);
        auto lengths = d_lengths.get_access<sycl_read>(cgh);
        auto offsets = d_offsets.get_access<sycl_read>(cgh);
        auto indexs = d_indexs.get_access<sycl_discard_write>(cgh);
        auto orders = d_orders.get_access<sycl_read>(cgh);
        auto words = d_words.get_access<sycl_discard_write>(cgh);
        auto magicBase = d_magicBase.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class index7> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
            kernel_createIndex7(
                reads.get_pointer(), 
                lengths.get_pointer(),
                offsets.get_pointer(), 
                indexs.get_pointer(),
                orders.get_pointer(),
                words.get_pointer(), 
                magicBase.get_pointer(), 
                readsCount,
                item);
        });
      });
      break;
  }

  // createCutoff(data, option);  // create threshold ok
  float threshold = option.threshold;
  buffer<int, 1> d_wordCutoff (readsCount);
  q.submit([&](handler &cgh) {
    auto reads = d_reads.get_access<sycl_read>(cgh);
    auto lengths = d_lengths.get_access<sycl_read>(cgh);
    auto words = d_words.get_access<sycl_read>(cgh);
    auto wordCutoff = d_wordCutoff.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class index7> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
        kernel_createCutoff(threshold, 
            wordLength, 
            lengths.get_pointer(),
            words.get_pointer(),
            wordCutoff.get_pointer(),
            readsCount, 
            item);
    });
  });

  // sortIndex(data);  // sort index ok
  q.submit([&](handler &cgh) {
    auto indexs = d_indexs.get_access<sycl_read>(cgh);
    cgh.copy(indexs, h_indexs);
  });
  q.submit([&](handler &cgh) {
    auto offsets = d_offsets.get_access<sycl_read>(cgh);
    cgh.copy(offsets, h_offsets);
  });
  q.submit([&](handler &cgh) {
    auto words = d_words.get_access<sycl_read>(cgh);
    cgh.copy(words, h_words);
  });
  q.wait();

  for (int i = 0; i< readsCount; i++) {
    int start = h_offsets[i];
    int length = h_words[i];
    std::sort(&h_indexs[start], &h_indexs[start]+length);
  }

  // mergeIndex(data);  // merge index ok
  q.submit([&](handler &cgh) {
    auto indexs = d_indexs.get_access<sycl_discard_write>(cgh);
    cgh.copy(h_indexs, indexs);
  });

  q.submit([&](handler &cgh) {
    auto indexs = d_indexs.get_access<sycl_read>(cgh);
    auto offsets = d_offsets.get_access<sycl_read>(cgh);
    auto words = d_words.get_access<sycl_read>(cgh);
    auto orders = d_orders.get_access<sycl_write>(cgh);
    cgh.parallel_for<class mergeIndex> (nd_range<1>(index_gws, index_lws), [=] (nd_item<1> item) {
        kernel_mergeIndex(offsets.get_pointer(), 
            indexs.get_pointer(), 
            orders.get_pointer(),
            words.get_pointer(), 
            readsCount, 
            item);
    });
  });

  // clustering(option, data, bench);  // clustering ok
  int* h_cluster = (int*) malloc (sizeof(int) * readsCount);
  for (int i = 0; i < readsCount; i++) {
    h_cluster[i] = -1;
  }

  buffer<int, 1> d_cluster(h_cluster, readsCount);

  unsigned short* table = (unsigned short*) malloc (sizeof(unsigned short) * 65536);
  memset(table, 0, 65536*sizeof(unsigned short));  // fill zero
  buffer<unsigned short, 1> d_table(table, 65536);

  int r = -1; // 

  range<1> makeTable_gws(128*128);
  range<1> makeTable_lws(128);
  range<1> cleanTable_gws(128*128);
  range<1> cleanTable_lws(128);
  range<1> magic_gws((readsCount+127)/128*128);
  range<1> magic_lws(128);
  range<1> filter_gws((readsCount+127)/128*128);
  range<1> filter_lws(128);
  range<1> align_gws((readsCount+127)/128*128);
  range<1> align_lws(128);

  while (r < readsCount) {  // clustering

    updateRepresentative(q, d_cluster, &r, readsCount);  // update representative
    if (r >= readsCount-1) {  // complete
      break;
    }
    std::cout << r << "/" << readsCount << std::endl;

    q.submit([&](sycl::handler &cgh) {
      auto offsets = d_offsets.get_access<sycl_read>(cgh);
      auto indexs = d_indexs.get_access<sycl_read>(cgh);
      auto orders = d_orders.get_access<sycl_read>(cgh);
      auto words = d_words.get_access<sycl_read>(cgh);
      auto table = d_table.get_access<sycl_write>(cgh);
      cgh.parallel_for<class makeTable>(nd_range<1>(makeTable_gws, makeTable_lws), [=] (nd_item<1> item) {
            kernel_makeTable(offsets.get_pointer(), indexs.get_pointer(), orders.get_pointer(),
                words.get_pointer(), table.get_pointer(), r, item);
      });
    }); // create table

    q.submit([&](sycl::handler &cgh) {
      auto lengths = d_lengths.get_access<sycl_read>(cgh);
      auto magicBase = d_magicBase.get_access<sycl_read>(cgh);
      auto cluster = d_cluster.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class magic>(nd_range<1>(magic_gws, magic_lws), [=] (nd_item<1> item) {
            kernel_magic(threshold, lengths.get_pointer(), magicBase.get_pointer(),
                cluster.get_pointer(), r, readsCount, item);
      });
    }); // magic filter

    q.submit([&](sycl::handler &cgh) {
      auto lengths = d_lengths.get_access<sycl_read>(cgh);
      auto offsets = d_offsets.get_access<sycl_read>(cgh);
      auto indexs = d_indexs.get_access<sycl_read>(cgh);
      auto orders = d_orders.get_access<sycl_read>(cgh);
      auto words = d_words.get_access<sycl_read>(cgh);
      auto magicBase = d_magicBase.get_access<sycl_read>(cgh);
      auto cluster = d_cluster.get_access<sycl_read_write>(cgh);
      auto wordCutoff = d_wordCutoff.get_access<sycl_read>(cgh);
      auto table = d_table.get_access<sycl_read>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> result (128, cgh);
      cgh.parallel_for<class filter>(nd_range<1>(filter_gws, filter_lws), [=] (nd_item<1> item) {
          kernel_filter(threshold, wordLength, 
              lengths.get_pointer(),
              offsets.get_pointer(), indexs.get_pointer(), orders.get_pointer(), words.get_pointer(),
              wordCutoff.get_pointer(), cluster.get_pointer(), table.get_pointer(),
              readsCount,
              item, 
              result.get_pointer());
      });
    }); // word filter
    q.submit([&](sycl::handler &cgh) {
      auto lengths = d_lengths.get_access<sycl_read>(cgh);
      auto offsets = d_offsets.get_access<sycl_read>(cgh);
      auto compressed = d_compressed.get_access<sycl_read>(cgh);
      auto gaps = d_gaps.get_access<sycl_read>(cgh);
      auto cluster = d_cluster.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class align>(nd_range<1>(align_gws, align_lws), [=] (nd_item<1> item) {

          kernel_align(threshold, 
              lengths.get_pointer(), 
              offsets.get_pointer(),
              compressed.get_pointer(), 
              gaps.get_pointer(), 
              r,
              cluster.get_pointer(), 
              readsCount, item);
      });
    }); // dynamic programming

    q.submit([&](sycl::handler &cgh) {
      auto offsets = d_offsets.get_access<sycl_read>(cgh);
      auto indexs = d_indexs.get_access<sycl_read>(cgh);
      auto orders = d_orders.get_access<sycl_read>(cgh);
      auto words = d_words.get_access<sycl_read>(cgh);
      auto table = d_table.get_access<sycl_write>(cgh);
      cgh.parallel_for<class cleanTable>(nd_range<1>(cleanTable_gws, cleanTable_lws), [=] (nd_item<1> item) {
        kernel_cleanTable(offsets.get_pointer(), indexs.get_pointer(), orders.get_pointer(),
                words.get_pointer(), table.get_pointer(), r, item);
      }); // table fill zero
    });
  }

  q.submit([&](handler &cgh) {
    auto cluster = d_cluster.get_access<sycl_read>(cgh);
    cgh.copy(cluster, h_cluster);
  }).wait();

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
  std::cout << "cluster count：" << sum << std::endl;
  free(h_lengths);
  free(h_offsets);
  free(h_reads);
  free(h_indexs);
  free(h_words);
  free(h_cluster);
  free(table);

  return 0;
}
