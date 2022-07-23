
/*
   __kernel void pgain_kernel(
   __global Point_Struct *p,       
   __global float *coord_h,
   __global float * work_mem_h,      
   __global int *center_table,
   __global char *switch_membership,      
   __local float *coord_s_acc,
   int num,
   int dim,
   long x,
   int K){  
   */

/* block ID and global thread ID */
const int local_id = omp_get_thread_num();
const int thread_id = omp_get_team_num()*omp_get_num_threads()+local_id;

if(thread_id<num){
  // coordinate mapping of point[x] to shared mem
  if(local_id == 0)
    for(int i=0; i<dim; i++){ 
      coord_s_acc[i] = coord_h[i*num + x];
    }
  #pragma omp barrier

  // cost between this point and point[x]: euclidean distance multiplied by weight
  float x_cost = 0.0;
  for(int i=0; i<dim; i++)
    x_cost += (coord_h[(i*num)+thread_id]-coord_s_acc[i]) * (coord_h[(i*num)+thread_id]-coord_s_acc[i]);
  x_cost = x_cost * p_h[thread_id].weight;

  float current_cost = p_h[thread_id].cost;

  int base = thread_id*(K+1);   
  // if computed cost is less then original (it saves), mark it as to reassign    
  if ( x_cost < current_cost ){
    switch_membership[thread_id] = '1';
    int addr_1 = base + K;
    work_mem_h[addr_1] = x_cost - current_cost;
  }
  // if computed cost is larger, save the difference
  else {
    int assign = p_h[thread_id].assign;
    int addr_2 = base + center_table[assign];
    work_mem_h[addr_2] += current_cost - x_cost;
  }
}

