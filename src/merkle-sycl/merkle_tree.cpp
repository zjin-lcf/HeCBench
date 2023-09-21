#include "merkle_tree.hpp"

void
merklize_approach_1(sycl::queue& q,
                    const sycl::ulong* leaves,
                    sycl::ulong* const intermediates,
                    const size_t leaf_count,
                    const size_t wg_size,
                    const sycl::ulong4* mds,
                    const sycl::ulong4* ark1,
                    const sycl::ulong4* ark2)
{
  // ensure only working with powers of 2 -many leaves
  assert((leaf_count & (leaf_count - 1)) == 0);
  // checking that requested work group size for first
  // phase of kernel dispatch is valid
  //
  // for next rounds of kernel dispatches, work group
  // size will be adapted when required !
  assert(wg_size <= (leaf_count >> 1));

  const size_t output_offset = leaf_count >> 1;

  // this is first phase of kernel dispatch, where I compute
  // ( in parallel ) all intermediate nodes just above leaves of tree
  sycl::event evt_0 = q.submit([&](sycl::handler& h) {
    h.parallel_for<class kernelMerklizeRescuePrimeApproach1Phase0>(
      sycl::nd_range<1>{ sycl::range<1>{ output_offset },
                         sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        merge(leaves + idx * (DIGEST_SIZE >> 1),
              intermediates + (output_offset + idx) * DIGEST_SIZE,
              mds,
              ark1,
              ark2);
      });
  });

  // for computing all remaining intermediate nodes, we'll need to
  // dispatch `rounds` -many kernels, where each round is data dependent
  // on just previous one
  const size_t rounds =
    static_cast<size_t>(sycl::log2(static_cast<double>(leaf_count >> 1)));

  std::vector<sycl::event> evts;
  evts.reserve(rounds);

  for (size_t r = 0; r < rounds; r++) {
    sycl::event evt = q.submit([&](sycl::handler& h) {
      if (r == 0) {
        h.depends_on(evt_0);
      } else {
        h.depends_on(evts.at(r - 1));
      }

      // these many intermediate nodes to be computed during this
      // kernel dispatch round
      const size_t offset = leaf_count >> (r + 2);

      h.parallel_for<class kernelMerklizeRescuePrimeApproach1Phase1>(
        sycl::nd_range<1>{
          sycl::range<1>{ offset },
          sycl::range<1>{ offset < wg_size ? offset : wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t idx = it.get_global_linear_id();

          merge(intermediates + (offset << 1) * DIGEST_SIZE +
                  idx * (DIGEST_SIZE >> 1),
                intermediates + (offset + idx) * DIGEST_SIZE,
                mds,
                ark1,
                ark2);
        });
    });
    evts.push_back(evt);
  }

  evts.at(rounds - 1).wait();
}

