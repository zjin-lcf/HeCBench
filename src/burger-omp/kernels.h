#define idx(i,j)   (i)*x_points+(j)

// begin of core
void core (
    const int numTeams,
    const int numThreads,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const double *__restrict__ u,
    const double *__restrict__ v,
    const int x_points,
    const int y_points,
    const double nu,
    const double del_t,
    const double del_x,
    const double del_y)
{
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for(int i = 1; i < y_points-1; i++){
    for(int j = 1; j < x_points-1; j++){
      u_new[idx(i,j)] = u[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (u[idx(i,j+1)] + u[idx(i,j-1)] - 2 * u[idx(i,j)]) + 
                                      (nu*del_t/(del_y*del_y)) * (u[idx(i+1,j)] + u[idx(i-1,j)] - 2 * u[idx(i,j)]) - 
                                              (del_t/del_x)*u[idx(i,j)] * (u[idx(i,j)] - u[idx(i,j-1)]) - 
                                              (del_t/del_y)*v[idx(i,j)] * (u[idx(i,j)] - u[idx(i-1,j)]);

      v_new[idx(i,j)] = v[idx(i,j)] + (nu*del_t/(del_x*del_x)) * (v[idx(i,j+1)] + v[idx(i,j-1)] - 2 * v[idx(i,j)]) + 
                                      (nu*del_t/(del_y*del_y)) * (v[idx(i+1,j)] + v[idx(i-1,j)] - 2 * v[idx(i,j)]) -
                                                (del_t/del_x)*u[idx(i,j)] * (v[idx(i,j)] - v[idx(i,j-1)]) - 
                                                (del_t/del_y)*v[idx(i,j)] * (v[idx(i,j)] - v[idx(i-1,j)]);
    }
  }
}
// end of core


void bound_h (
    const int numTeams,
    const int numThreads,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for(int i = 0; i < x_points; i++){
    u_new[idx(0,i)] = 1.0;
    v_new[idx(0,i)] = 1.0;
    u_new[idx(y_points-1,i)] = 1.0;
    v_new[idx(y_points-1,i)] = 1.0;
  }
}

void bound_v (
    const int numTeams,
    const int numThreads,
    double *__restrict__ u_new,
    double *__restrict__ v_new,
    const int x_points,
    const int y_points)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for(int j = 0; j < y_points; j++){
    u_new[idx(j,0)] = 1.0;
    v_new[idx(j,0)] = 1.0;
    u_new[idx(j,x_points-1)] = 1.0;
    v_new[idx(j,x_points-1)] = 1.0;
  }
}

void update (
    const int numTeams,
    const int numThreads,
    double *__restrict__ u,
    double *__restrict__ v,
    const double *__restrict__ u_new,
    const double *__restrict__ v_new,
    const int n)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for(int i = 0; i < n; i++){
    u[i] = u_new[i];
    v[i] = v_new[i];
  }
}
