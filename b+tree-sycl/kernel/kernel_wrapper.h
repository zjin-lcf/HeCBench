#include "common.h"
void 
kernel_wrapper(	queue &q,
		record *records,
		long records_mem,
		knode *knodes,
		long knodes_elem,
		long knodes_mem,

		int order,
		long maxheight,
		int count,

		long *currKnode,
		long *offset,
		int *keys,
		record *ans);
