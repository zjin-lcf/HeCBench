#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <omp.h>

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
  }
  while(stop);
}
//----------------------------------------------------------
//--breadth first search on GPUs
//----------------------------------------------------------
void run_bfs_gpu(int no_of_nodes, Node *d_graph_nodes, int edge_list_size, \
    int *d_graph_edges, char *d_graph_mask, char *d_updating_graph_mask, \
    char *d_graph_visited, int *d_cost) throw(std::string)
{
  char d_over[1];

#pragma omp target data map(to: d_graph_nodes[0:no_of_nodes], \
                                d_graph_edges[0:edge_list_size], \
                                d_graph_visited[0:no_of_nodes], \
                                d_graph_mask[0:no_of_nodes], \
                                d_updating_graph_mask[0:no_of_nodes]) \
                        map(alloc: d_over[0:1])\
                        map(tofrom: d_cost[0:no_of_nodes])
  {
    long time = 0;
    do {
      d_over[0] = 0;
      #pragma omp target update to (d_over[0:1])

      auto start = std::chrono::steady_clock::now();

      #pragma omp target teams distribute parallel for thread_limit(MAX_THREADS_PER_BLOCK)
      for (int tid = 0; tid < no_of_nodes; tid++) {
        if(d_graph_mask[tid]){
          d_graph_mask[tid]=0;
          const int num_edges = d_graph_nodes[tid].no_of_edges;
          const int starting = d_graph_nodes[tid].starting;

          for(int i=starting; i<(num_edges + starting); i++) {
            int id = d_graph_edges[i];
            if(!d_graph_visited[id]){
              d_cost[id]=d_cost[tid]+1;
              d_updating_graph_mask[id]=1;
            }
          }
        }  
      }

      #pragma omp target teams distribute parallel for thread_limit(MAX_THREADS_PER_BLOCK) 
      for (int tid = 0; tid < no_of_nodes; tid++) {
        if(d_updating_graph_mask[tid]){
          d_graph_mask[tid]=1;
          d_graph_visited[tid]=1;
          d_over[0]=1;
          d_updating_graph_mask[tid]=0;
        }
      }

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      #pragma omp target update from (d_over[0:1])

    } while (d_over[0]);

    printf("Total kernel execution time : %f (us)\n", time * 1e-3f);
  }
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
  int  *h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
  int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
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
