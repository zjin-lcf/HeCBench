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
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  #pragma omp target teams num_teams((n+BLOCK_SIZE-1)/BLOCK_SIZE) thread_limit(BLOCK_SIZE)
  {
    int local_cache[BLOCK_SIZE];
    #pragma omp parallel
    {
      const int local_id = omp_get_thread_num();
      const int local_size = BLOCK_SIZE;
      const int nid = omp_get_team_num();
      const int global_id = nid * local_size + local_id;
      if (global_id < n)
        //Copy values into local memory
        local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0;
      else
        local_cache[local_id] = 0;

      //Wait for all threads
      #pragma omp barrier

      for (int i = local_size/2; i > 0; i /= 2) {
        if(local_id < i) {
          local_cache[local_id] += local_cache[local_id + i];
        }
        //Wait for all threads
        #pragma omp barrier
      }

      if (local_id == 0) {
        // Update damage and n_neigh
        int neighbours = local_cache[0];
        n_neigh[nid] = neighbours;
        damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
      }
    }
  }
}

void damage_of_node_optimized(
  const int m,
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  #pragma omp target teams distribute num_teams(m) 
  for (int nid = 0; nid < m; nid++) {
    int lb = nid * BLOCK_SIZE;
    int ub = (lb + BLOCK_SIZE < n) ? lb + BLOCK_SIZE : n;

    int sum = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(BLOCK_SIZE)
    for (int i = lb; i < ub; i++)
      sum += nlist[i] != -1 ? 1 : 0;

    n_neigh[nid] = sum;
    damage[nid] = 1.0 - (double) sum / (double) (family[nid]);
  }
}
