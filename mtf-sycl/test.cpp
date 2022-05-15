#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <vector>
int main(){
  std::vector<int> vec( 1000 );
  std::fill(oneapi::dpl::execution::dpcpp_default, vec.begin(), vec.end(), 42);
  // each element of vec equals to 42
  return 0;
}
