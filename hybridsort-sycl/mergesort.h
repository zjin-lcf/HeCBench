#ifndef __MERGESORT
#define __MERGESORT

#include "bucketsort.h"

float4* runMergeSort(queue &q, int listsize, int divisions,
					 float4 *d_origList, float4 *d_resultList,
					 int *sizes, int *nullElements,
					 unsigned int *origOffsets);
#endif
