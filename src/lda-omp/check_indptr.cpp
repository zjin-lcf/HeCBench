#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int i;
  srand(123);
  
  const int num_words = 10266;
  const int block_cnt = 500;
  const int num_indptr = block_cnt;
  
  std::vector<int> indptr(num_indptr+1, 0);
  indptr[num_indptr] = num_words-1;
  for (i = num_indptr; i >= 1; i--) {
    int t = indptr[i] - 1 - (rand() % (num_words/num_indptr));
    if (t < 0) break;
    indptr[i-1] = t;
  }
  
  printf("indptr array (first 20):\n");
  for (i = 0; i <= 20 && i <= num_indptr; i++) {
    printf("indptr[%d] = %d\n", i, indptr[i]);
  }
  
  printf("\nindptr array (last 20):\n");
  for (i = num_indptr - 19; i <= num_indptr; i++) {
    if (i >= 0) printf("indptr[%d] = %d\n", i, indptr[i]);
  }
  
  printf("\nDocument sizes:\n");
  int zero_count = 0, negative_count = 0;
  for (i = 0; i < num_indptr; i++) {
    int doc_size = indptr[i+1] - indptr[i];
    if (doc_size == 0) zero_count++;
    if (doc_size < 0) {
      negative_count++;
      printf("DOC %d: size %d (beg=%d, end=%d)\n", i, doc_size, indptr[i], indptr[i+1]);
    }
  }
  
  printf("\nSummary:\n");
  printf("Total documents: %d\n", num_indptr);
  printf("Documents with 0 words: %d\n", zero_count);
  printf("Documents with negative size: %d\n", negative_count);
  
  return 0;
}
