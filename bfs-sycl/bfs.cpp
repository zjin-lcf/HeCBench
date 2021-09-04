//--by Jianbin Fang

#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include "common.h"
#include "util.h"

#define MAX_THREADS_PER_BLOCK 256


//Structure to hold a node information
struct Node
{
  int starting;
  int no_of_edges;
};


//----------------------------------------------------------
//--bfs on cpu
//--programmer:  jianbin
//--date:  26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
    char *h_graph_visited, int *h_cost_ref){
  char stop;
  int k = 0;
  do{
    //if no thread changes this value then the loop stops
    stop=0;
    for(int tid = 0; tid < no_of_nodes; tid++ )
    {
      if (h_graph_mask[tid] == 1){ 
        h_graph_mask[tid]=0;
        for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
          int id = h_graph_edges[i];  //--cambine: node id is connected with node tid
          if(!h_graph_visited[id]){  //--cambine: if node id has not been visited, enter the body below
            h_cost_ref[id]=h_cost_ref[tid]+1;
            h_updating_graph_mask[id]=1;
          }
        }
      }    
    }

    for(int tid=0; tid< no_of_nodes ; tid++ )
    {
      if (h_updating_graph_mask[tid] == 1){
        h_graph_mask[tid]=1;
        h_graph_visited[tid]=1;
        stop=1;
        h_updating_graph_mask[tid]=0;
      }
    }
    k++;
  }
  while(stop);
}
//----------------------------------------------------------
//--breadth first search on GPUs
//----------------------------------------------------------
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size,
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask,
    char *h_graph_visited, int *h_cost) noexcept(false) {
  char h_over;

  { // SYCL scopde
#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    const property_list props = property::buffer::use_host_ptr();
    buffer<Node,1> d_graph_nodes(h_graph_nodes, no_of_nodes, props);
    buffer<int,1> d_graph_edges(h_graph_edges, edge_list_size, props);
    buffer<char,1> d_graph_mask(h_graph_mask, no_of_nodes, props);
    buffer<char,1> d_updating_graph_mask(h_updating_graph_mask, no_of_nodes, props);
    buffer<char,1> d_graph_visited(h_graph_visited, no_of_nodes, props);
    buffer<int,1> d_cost(h_cost, no_of_nodes, props);

    d_graph_mask.set_final_data(nullptr);
    d_updating_graph_mask.set_final_data(nullptr);
    d_graph_visited.set_final_data(nullptr);

    buffer<char,1> d_over(1);

    int global_work_size = (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK * MAX_THREADS_PER_BLOCK;
    range<1> gws (global_work_size);
    range<1> lws (MAX_THREADS_PER_BLOCK);

    // invoke kernel
    do {
      h_over = 0;
      q.submit([&](handler& cgh) {
        auto d_over_acc = d_over.get_access<sycl_write>(cgh);
        cgh.copy(&h_over, d_over_acc);
      });

      q.submit([&](handler& cgh) {
        auto d_graph_nodes_acc = d_graph_nodes.get_access<sycl_read>(cgh);
        auto d_graph_edges_acc = d_graph_edges.get_access<sycl_read>(cgh);
        auto d_graph_mask_acc = d_graph_mask.get_access<sycl_write>(cgh);
        auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<sycl_write>(cgh);
        auto d_graph_visited_acc = d_graph_visited.get_access<sycl_read>(cgh);
        auto d_cost_acc = d_cost.get_access<sycl_write>(cgh);

        cgh.parallel_for<class kernel1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int tid = item.get_global_id(0);
          if (tid<no_of_nodes && d_graph_mask_acc[tid]) {
            d_graph_mask_acc[tid]=0;
            for(int i=d_graph_nodes_acc[tid].starting; 
                    i<(d_graph_nodes_acc[tid].no_of_edges + d_graph_nodes_acc[tid].starting); i++){
              int id = d_graph_edges_acc[i];
              if(!d_graph_visited_acc[id]){
                d_cost_acc[id]=d_cost_acc[tid]+1;
                d_updating_graph_mask_acc[id]=1;
              }
            }
          }    
        });
      });

      //--kernel 1

      q.submit([&](handler& cgh) {
        auto d_graph_mask_acc = d_graph_mask.get_access<sycl_write>(cgh);
        auto d_updating_graph_mask_acc = d_updating_graph_mask.get_access<sycl_read_write>(cgh);
        auto d_graph_visited_acc = d_graph_visited.get_access<sycl_write>(cgh);
        auto d_over_acc = d_over.get_access<sycl_write>(cgh);
        cgh.parallel_for<class kernel2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          int tid = item.get_global_id(0);
          if (tid<no_of_nodes && d_updating_graph_mask_acc[tid]) {
            d_graph_mask_acc[tid]=1;
            d_graph_visited_acc[tid]=1;
            d_over_acc[0]=1;
            d_updating_graph_mask_acc[tid]=0;
          }
        });
      });

      q.submit([&](handler& cgh) {
        auto d_over_acc = d_over.get_access<sycl_read>(cgh);
        cgh.copy(d_over_acc, &h_over);
      }).wait();
    }
    while (h_over);

  } // SYCL scope
}
void Usage(int argc, char**argv){

  fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
//----------------------------------------------------------
//--cambine:  main function
//--author:    created by Jianbin Fang
//--date:    25/01/2011
//----------------------------------------------------------
int main(int argc, char * argv[])
{
  int no_of_nodes;
  int edge_list_size;
  FILE *fp;
  Node* h_graph_nodes;
  char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
  char *input_f;
  if(argc!=2){
    Usage(argc, argv);
    exit(0);
  }

  input_f = argv[1];
  printf("Reading File\n");
  //Read in Graph from a file
  fp = fopen(input_f,"r");
  if(!fp){
    printf("Error Reading graph file %s\n", input_f);
    return 1;
  }

  int source = 0;

  fscanf(fp,"%d",&no_of_nodes);

  // allocate host memory
  h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
  h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
  h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
  h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);

  int start, edgeno;   
  // initalize the memory
  for(int i = 0; i < no_of_nodes; i++){
    fscanf(fp,"%d %d",&start,&edgeno);
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].no_of_edges = edgeno;
    h_graph_mask[i]=0;
    h_updating_graph_mask[i]=0;
    h_graph_visited[i]=0;
  }
  //read the source node from the file
  fscanf(fp,"%d",&source);
  source=0;
  //set the source node as 1 in the mask
  h_graph_mask[source]=1;
  h_graph_visited[source]=1;
  fscanf(fp,"%d",&edge_list_size);
  int id,cost;
  int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
  for(int i=0; i < edge_list_size ; i++){
    fscanf(fp,"%d",&id);
    fscanf(fp,"%d",&cost);
    h_graph_edges[i] = id;
  }

  if(fp) fclose(fp);    

  // allocate mem for the result on host side
  int *h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
  int *h_cost_ref = (int*) malloc(sizeof(int)*no_of_nodes);
  for(int i=0;i<no_of_nodes;i++){
    h_cost[i]=-1;
    h_cost_ref[i] = -1;
  }
  h_cost[source]=0;
  h_cost_ref[source]=0;    

  printf("run bfs (#nodes = %d) on device\n", no_of_nodes);
  run_bfs_gpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, 
      h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);  

  printf("run bfs (#nodes = %d) on host (cpu) \n", no_of_nodes);
  // initalize the memory again
  for(int i = 0; i < no_of_nodes; i++){
    h_graph_mask[i]=0;
    h_updating_graph_mask[i]=0;
    h_graph_visited[i]=0;
  }

  //set the source node as 1 in the mask
  source=0;
  h_graph_mask[source]=1;
  h_graph_visited[source]=1;
  run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, 
      h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);

  // verify
  compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
  free(h_graph_nodes);
  free(h_graph_mask);
  free(h_updating_graph_mask);
  free(h_graph_visited);
  free(h_cost);
  free(h_cost_ref);
  return 0;
}
