#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel.cpp"
#include "./kernel_wrapper.h"				// (in current directory)

void 
kernel_wrapper(record *records,
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
						record *ans)
{

	long long offload_start = get_time();

	int numBlocks;
	numBlocks = count;									// max # of blocks can be 65,535
	int threadsPerBlock;
	threadsPerBlock = order < 256 ? order : 256;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

	//==================================================50
	//	recordsD
	//==================================================50

	record *recordsD;
  recordsD =
      (record *)sycl::malloc_device(records_mem, dpct::get_default_queue());

        //==================================================50
	//	knodesD
	//==================================================50

	knode *knodesD;
  knodesD = (knode *)sycl::malloc_device(knodes_mem, dpct::get_default_queue());

        //==================================================50
	//	currKnodeD
	//==================================================50

	long *currKnodeD;
  currKnodeD = sycl::malloc_device<long>(count, dpct::get_default_queue());

        //==================================================50
	//	offsetD
	//==================================================50

	long *offsetD;
  offsetD = sycl::malloc_device<long>(count, dpct::get_default_queue());

        //==================================================50
	//	keysD
	//==================================================50

	int *keysD;
  keysD = sycl::malloc_device<int>(count, dpct::get_default_queue());

        //==================================================50
	//	ansD
	//==================================================50

	record *ansD;
  ansD = sycl::malloc_device<record>(count, dpct::get_default_queue());

        //==================================================50
	//	recordsD
	//==================================================50

  dpct::get_default_queue().memcpy(recordsD, records, records_mem);

        //==================================================50
	//	knodesD
	//==================================================50

  dpct::get_default_queue().memcpy(knodesD, knodes, knodes_mem);

        //==================================================50
	//	currKnodeD
	//==================================================50

  dpct::get_default_queue().memcpy(currKnodeD, currKnode, count * sizeof(long));

        //==================================================50
	//	offsetD
	//==================================================50

  dpct::get_default_queue().memcpy(offsetD, offset, count * sizeof(long));

        //==================================================50
	//	keysD
	//==================================================50

  dpct::get_default_queue().memcpy(keysD, keys, count * sizeof(int));

        //==================================================50
	//	ansD
	//==================================================50

  dpct::get_default_queue().memcpy(ansD, ans, count * sizeof(record));

        //======================================================================================================================================================150
	// findK kernel
	//======================================================================================================================================================150

  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          findK(maxheight, knodesD, knodes_elem, recordsD, currKnodeD, offsetD,
                keysD, ansD, item_ct1);
        });
  });

        //==================================================50
	//	ansD
	//==================================================50

  dpct::get_default_queue().memcpy(ans, ansD, count * sizeof(record)).wait();

  sycl::free(recordsD, dpct::get_default_queue());
  sycl::free(knodesD, dpct::get_default_queue());
  sycl::free(currKnodeD, dpct::get_default_queue());
  sycl::free(offsetD, dpct::get_default_queue());
  sycl::free(keysD, dpct::get_default_queue());
  sycl::free(ansD, dpct::get_default_queue());

        long long offload_end = get_time();

	printf("Total time:\n");
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000); 
}
