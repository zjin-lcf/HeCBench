/* Calculate the damage of each node.
 *
 * nlist - An (n, local_size) array containing the neighbour lists,
 *         a value of -1 corresponds to a broken bond.
 * family - An array of the initial number of neighbours for each node.
 * n_neigh - An array of the number of neighbours (particles bound) for each node.
 * damage - An array of the damage for each node. 
 */
void damage_of_node(
  const int numTeams,
  const int numThreads,
  const int n,
  const int *__restrict nlist,
  const int *__restrict family,
        int *__restrict n_neigh,
     double *__restrict damage)
{
  #pragma omp target teams distribute num_teams(numTeams)
  for (int nid = 0; nid < numTeams; nid++)
  {
    int neighbours = 0;
    #pragma omp parallel for reduction(+ : neighbours) num_threads(numThreads)
    for (int local_id = 0 ; local_id < numThreads ; ++ local_id) {
      int global_id = nid * numThreads + local_id;
      if (global_id < n && nlist[global_id] != -1) {
        neighbours++;
      }
    }
    n_neigh[nid] = neighbours ;
    damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
  }
}
