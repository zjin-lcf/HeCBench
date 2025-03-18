/* Calculate the damage of each node.
 *
 * nlist - An (n, local_size) array containing the neighbour lists,
 *         a value of -1 corresponds to a broken bond.
 * family - An array of the initial number of neighbours for each node.
 * n_neigh - An array of the number of neighbours (particles bound) for each node.
 * damage - An array of the damage for each node.
 * local_cache - local (local_size) array to store the bond breakages.
 */
void damage_of_node(
  sycl::queue &q,
  sycl::range<3> &gws,
  sycl::range<3> &lws,
  const int slm_size,
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  auto cgf = [&](sycl::handler& cgh) {
    sycl::local_accessor<int, 1> local_cache (sycl::range<1>(slm_size), cgh);
    auto kfn = [=](sycl::nd_item<3> item) {
      const int global_id = item.get_global_id(2);
      if (global_id >= n) return;

      const int local_id = item.get_local_id(2);
      const int local_size = item.get_local_range(2);

      //Copy values into local memory
      local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0;

      //Wait for all threads
      item.barrier(sycl::access::fence_space::local_space);

      for (int i = local_size/2; i > 0; i /= 2) {
        if(local_id < i){
          local_cache[local_id] += local_cache[local_id + i];
        }
        //Wait for all threads
        item.barrier(sycl::access::fence_space::local_space);
      }

      if (local_id == 0) {
        //Get the reduced damages
        int nid = item.get_group(2);
        // Update damage and n_neigh
        int neighbours = local_cache[0];
        n_neigh[nid] = neighbours;
        damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
