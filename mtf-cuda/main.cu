#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/find.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

void mtf(std::vector<char> &word, bool output)
{
  thrust::device_vector<char> d_list(256);
  thrust::host_vector<char> list(256);
  thrust::device_vector<char> d_word(word.size());

  size_t counter;
  thrust::device_vector<char>::iterator iter, count;
  thrust::host_vector<char> h_word(word.size());
  h_word = word;
  d_word = h_word;

  for (counter = 0; counter < word.size(); counter++)
  {
    thrust::copy(list.begin(), list.end(), d_list.begin());

    h_word[0] = d_word[counter];
    iter = thrust::find(d_list.begin(), d_list.end(), d_word[counter]);

    if (d_list[0] != h_word[0])
    {
      thrust::copy(d_list.begin(), iter, list.begin()+1);
      list[0] = h_word[0];
    }
  }

  thrust::copy(list.begin(), list.end(), d_list.begin());
  thrust::copy(word.begin(), word.end(), d_word.begin());
  for (counter = 0; counter < list.size(); counter++)
  {
    iter = thrust::find(d_word.begin(), d_word.end(), d_list[counter]);
    while (iter != d_word.end())
    {
      *iter = counter;
      iter = thrust::find(d_word.begin(), d_word.end(), d_list[counter]);
    }
  }
  thrust::copy(d_word.begin(), d_word.end(), h_word.begin());

  if (output) {
    for (counter = 0; counter < word.size(); counter++)
      printf("%d ", h_word[counter]);
    printf("\n");
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    printf("Usage: %s <string_input> <repeat>\n", argv[0]);
    exit(1);
  }

  const int len = strlen(argv[1]);
  std::vector<char> word(argv[1], argv[1] + len);

  const int repeat = atoi(argv[2]);

  // output MTF result
  mtf(word, true);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) mtf(word, false);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time: " << (time * 1e-9f) / repeat << " (s)\n";
  return 0;
}
