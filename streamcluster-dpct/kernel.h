#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void compute_cost(const Point_Struct *p_d_acc, const float *coord_d_acc,
                  float *work_mem_d_acc, const int *center_table_d_acc,
                  char *switch_membership_d_acc, const int num, const int dim,
                  const long x, const int K, sycl::nd_item<3> item_ct1,
                  uint8_t *dpct_local)
{
  auto coord_s_acc = (float *)dpct_local;
  /* block ID and global thread ID */
  const int local_id = item_ct1.get_local_id(2);
  const int thread_id =
      item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + local_id;

  if(thread_id<num){
    // coordinate mapping of point[x] to shared mem
    if(local_id == 0)
      for(int i=0; i<dim; i++){ 
        coord_s_acc[i] = coord_d_acc[i*num + x];
      }
    item_ct1.barrier();

    // cost between this point and point[x]: euclidean distance multiplied by weight
    float x_cost = 0.0;
    for(int i=0; i<dim; i++)
      x_cost += (coord_d_acc[(i*num)+thread_id]-coord_s_acc[i]) * (coord_d_acc[(i*num)+thread_id]-coord_s_acc[i]);
    x_cost = x_cost * p_d_acc[thread_id].weight;

    float current_cost = p_d_acc[thread_id].cost;

    int base = thread_id*(K+1);   
    // if computed cost is less then original (it saves), mark it as to reassign    
    if ( x_cost < current_cost ){
      switch_membership_d_acc[thread_id] = '1';
      int addr_1 = base + K;
      work_mem_d_acc[addr_1] = x_cost - current_cost;
    }
    // if computed cost is larger, save the difference
    else {
      int assign = p_d_acc[thread_id].assign;
      int addr_2 = base + center_table_d_acc[assign];
      work_mem_d_acc[addr_2] += current_cost - x_cost;
    }
  }
}

