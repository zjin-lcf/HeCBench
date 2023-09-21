#ifndef KERNEL_WRAPPER_H
#define KERNEL_WRAPPER_H

#include <sycl/sycl.hpp>
void 
kernel_wrapper(	sycl::queue &q,
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
#endif
