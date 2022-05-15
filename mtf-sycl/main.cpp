#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <CL/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

void mtf(sycl::queue q, std::vector<char> word, bool output)
{
  auto policy = oneapi::dpl::execution::make_device_policy(q);

  std::vector<char> d_list(256);

  std::vector<char> list(256);
  
  std::vector<char> d_word (word.size());

  size_t counter;
  std::vector<char> h_word(word.size());
  h_word = word;
  d_word = h_word;

  for (counter = 0; counter < word.size(); counter++)
  {
    std::copy(policy, list.begin(), list.end(), d_list.begin());

    h_word[0] = d_word[counter];

    auto iter = std::find(policy, d_list.begin(), d_list.end(), d_word[counter]);

    if (d_list[0] != h_word[0])
    {
      std::copy(policy, d_list.begin(), iter, list.begin() + 1);
      list[0] = h_word[0];
    }
  }

  std::copy(policy, list.begin(), list.end(), d_list.begin());
  std::copy(policy, word.begin(), word.end(), d_word.begin());
  for (counter = 0; counter < list.size(); counter++)
  {
    auto iter = std::find(policy, d_word.begin(), d_word.end(), d_list[counter]);
    while (iter != d_word.end())
    {
      *iter = counter;
      iter = std::find(policy, d_word.begin(), d_word.end(), d_list[counter]);
    }
  }
  std::copy(policy, d_word.begin(), d_word.end(), h_word.begin());

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

#ifdef USE_GPU
  sycl::gpu_selector dev_sel;
#else
  sycl::cpu_selector dev_sel;
#endif
  sycl::queue q(dev_sel);

  // output MTF result
  mtf(q, word, true);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mtf(q, word, false);

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  std::cout << "Total execution time: " << time.count() << std::endl;
  return 0;
}
