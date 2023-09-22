#ifndef KERNEL2_WRAPPER_H
#define KERNEL2_WRAPPER_H

#include <sycl/sycl.hpp>
void 
kernel2_wrapper(sycl::queue &q,
		knode *knodes,
		long knodes_elem,
		long knodes_mem,

		int order,
		long maxheight,
		int count,

		long *currKnode,
		long *offset,
		long *lastKnode,
		long *offset_2,
		int *start,
		int *end,
		int *recstart,
		int *reclength);
#endif
