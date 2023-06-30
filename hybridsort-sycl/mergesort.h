#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.h"

sycl::float4* runMergeSort(sycl::queue &q, int listsize, int divisions,
					 sycl::float4 *d_origList, sycl::float4 *d_resultList,
					 int *sizes, int *nullElements,
					 unsigned int *origOffsets);
#endif
