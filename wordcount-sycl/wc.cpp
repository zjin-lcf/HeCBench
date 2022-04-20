#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <numeric>  // C++ inner product
#include <vector>

// This example computes the number of words in a text sample.
// It counts the number of characters which start a new word, i.e.
// the number of characters where input[i] is an alphabetical
// character and input[i-1] is not an alphabetical character.

// determines whether the character is alphabetical
bool is_alpha(const char c)
{
  return (c >= 'A' && c <= 'z');
}

// determines whether the right character begins a new word
struct is_word_start
{
  bool operator()(const char& left, const char& right) const
  {
    return is_alpha(right) && !is_alpha(left);
  }
};

int word_count_reference(const std::vector<char> &input)
{
  // check for empty string
  if (input.empty()) return 0;

  // compute the number characters that start a new word
  int wc = std::inner_product(
      input.cbegin(), input.cend() - 1, // sequence of left characters
      input.cbegin() + 1,               // sequence of right characters
      0,                                // initialize sum to 0
      std::plus<int>(),                 // sum values together
      is_word_start());                 // how to compare the left and right characters

  // if the first character is alphabetical, then it also begins a word
  if (is_alpha(input.front())) wc++;
  
  return wc;
}

int word_count(const std::vector<char> &input)
{
  // check for empty string
  if (input.empty()) return 0;

  // compute the number characters that start a new word
  // https://github.com/oneapi-src/oneDPL/issues/570
  int wc = oneapi::dpl::transform_reduce(
           oneapi::dpl::execution::dpcpp_default,
           input.cbegin(), input.cend() - 1, // sequence of left characters
           input.cbegin() + 1,               // sequence of right characters
           0,                                // initialize sum to 0
           std::plus<int>(),                 // sum values together
           is_word_start());                 // how to compare the left and right characters

  // if the first character is alphabetical, then it also begins a word
  if (is_alpha(input.front())) wc++;
  
  return wc;
}

