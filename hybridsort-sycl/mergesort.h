#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.h"

cl::sycl::float4* runMergeSort(queue &q, int listsize, int divisions,
					 cl::sycl::float4 *d_origList, cl::sycl::float4 *d_resultList,
					 int *sizes, int *nullElements,
					 unsigned int *origOffsets);
#endif
