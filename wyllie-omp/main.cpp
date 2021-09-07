#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include "utils.h"

// kernel execution times
#define REPEAT 100

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: ./%s <list size> <0:an ordered list | otherwise: a random list>\n", argv[0]);
    exit(-1);
  }

  int elems = atoi(argv[1]);
  int setRandomList = atoi(argv[2]);
  int i;

  std::vector<int> next (elems);
  std::vector<int> rank (elems);
  std::vector<long> list (elems);
  std::vector<long> d_res (elems);
  std::vector<long> h_res (elems);

  // generate an array in which each element contains the index of the next element
  if (setRandomList)
    random_list(next);
  else
    ordered_list(next);

  // initialize the rank list
  for (i = 0; i < elems; i++) {
    rank[i] = next[i] == NIL ? 0 : 1;
  }

  // pack next and rank as a 64-bit number
  for (i = 0; i < elems; i++) list[i] = ((long)next[i] << 32) | rank[i];

  // run list ranking on a device
  long* plist = list.data();
  const int teams = (elems + 255) / 256; 

#pragma omp target data map (tofrom: plist[0:elems])
{
  for (i = 0; i < REPEAT; i++) {
    #pragma omp target update to (plist[0:elems])

    #pragma omp target teams num_teams(teams) thread_limit(256)
    {
      #pragma omp parallel
      {
        int index = omp_get_team_num() * 256 + omp_get_thread_num();
	if (index < elems) {
          long node, next;
          while ( ((node = plist[index]) >> 32) != NIL && 
                  ((next = plist[node >> 32]) >> 32) != NIL )
          {
            long temp = (node & MASK) ;
            temp += (next & MASK) ;
            temp += (next >> 32) << 32;
            #pragma omp barrier
            plist [ index ] = temp ;
          } 
        }
      }
    }
  }
}

  for (i = 0; i < elems; i++) d_res[i] = list[i] & MASK;

  // verify
  // compute distance from the *end* of the list (note the first element is the head node)
  h_res[0] = elems-1;
  i = 0;
  for (int r = 1; r < elems; r++) {
    h_res[next[i]] = elems-1-r;
    i = next[i];
  }
 
#ifdef DEBUG
  printf("Ranks:\n");
  for (i = 0; i < elems; i++) {
    printf("%d: %d %d\n", i, h_res[i], d_res[i]);
  }
#endif

  printf("%s\n", (h_res == d_res) ? "PASS" : "FAIL");
   
  return 0;
}
