#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel2.cpp"
#include "./kernel2_wrapper.h"

void 
kernel2_wrapper(	knode *knodes,
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
							int *reclength)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

        long long offload_start = get_time();

	//====================================================================================================100
	//	EXECUTION PARAMETERS
	//====================================================================================================100

	int numBlocks;
	numBlocks = count;
	int threadsPerBlock;
	threadsPerBlock = order < 256 ? order : 256;

	printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);


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
	//	lastKnodeD
	//==================================================50

	long *lastKnodeD;
  lastKnodeD = sycl::malloc_device<long>(count, dpct::get_default_queue());

        //==================================================50
	//	offset_2D
	//==================================================50

	long *offset_2D;
  offset_2D = sycl::malloc_device<long>(count, dpct::get_default_queue());

        //==================================================50
	//	startD
	//==================================================50

	int *startD;
  startD = sycl::malloc_device<int>(count, dpct::get_default_queue());

        //==================================================50
	//	endD
	//==================================================50

	int *endD;
  endD = sycl::malloc_device<int>(count, dpct::get_default_queue());

        //==================================================50
	//	ansDStart
	//==================================================50

	int *ansDStart;
  ansDStart = sycl::malloc_device<int>(count, dpct::get_default_queue());

        //==================================================50
	//	ansDLength
	//==================================================50

	int *ansDLength;
  ansDLength = sycl::malloc_device<int>(count, dpct::get_default_queue());

  dpct::get_default_queue().memcpy(knodesD, knodes, knodes_mem);

  dpct::get_default_queue().memcpy(currKnodeD, currKnode, count * sizeof(long));

  dpct::get_default_queue().memcpy(offsetD, offset, count * sizeof(long));

  dpct::get_default_queue().memcpy(lastKnodeD, lastKnode, count * sizeof(long));

  dpct::get_default_queue().memcpy(offset_2D, offset_2, count * sizeof(long));

  dpct::get_default_queue().memcpy(startD, start, count * sizeof(int));

  dpct::get_default_queue().memcpy(endD, end, count * sizeof(int));

  dpct::get_default_queue().memcpy(ansDStart, recstart, count * sizeof(int));

  dpct::get_default_queue().memcpy(ansDLength, reclength, count * sizeof(int));

        // [GPU] findRangeK kernel
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                              sycl::range<3>(1, 1, threadsPerBlock),
                          sycl::range<3>(1, 1, threadsPerBlock)),
        [=](sycl::nd_item<3> item_ct1) {
          findRangeK(maxheight, knodesD, knodes_elem, currKnodeD, offsetD,
                     lastKnodeD, offset_2D, startD, endD, ansDStart, ansDLength,
                     item_ct1);
        });
  });

  dpct::get_default_queue().memcpy(recstart, ansDStart, count * sizeof(int));

  dpct::get_default_queue().memcpy(reclength, ansDLength, count * sizeof(int));

  dpct::get_current_device().queues_wait_and_throw();

  sycl::free(knodesD, dpct::get_default_queue());
  sycl::free(currKnodeD, dpct::get_default_queue());
  sycl::free(offsetD, dpct::get_default_queue());
  sycl::free(lastKnodeD, dpct::get_default_queue());
  sycl::free(offset_2D, dpct::get_default_queue());
  sycl::free(startD, dpct::get_default_queue());
  sycl::free(endD, dpct::get_default_queue());
  sycl::free(ansDStart, dpct::get_default_queue());
  sycl::free(ansDLength, dpct::get_default_queue());

        long long offload_end = get_time();

	printf("Total time:\n"); 
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000); 
}

