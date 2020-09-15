#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#define MAX_THREADS_PER_BLOCK 256

#ifdef  PROFILING
#include "timer.h"
#endif

#include "util.h"

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

__global__ void
Kernel( const Node* d_graph_nodes, const int* d_graph_edges, char* d_graph_mask, char* d_updatind_graph_mask, const char *d_graph_visited, int* d_cost, const int no_of_nodes) 
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if( tid<no_of_nodes && d_graph_mask[tid])
	{
		d_graph_mask[tid]=0;
		for(int i=d_graph_nodes[tid].starting; i<(d_graph_nodes[tid].no_of_edges + d_graph_nodes[tid].starting); i++)
			{
			int id = d_graph_edges[i];
			if(!d_graph_visited[id])
				{
				d_cost[id]=d_cost[tid]+1;
				d_updatind_graph_mask[id]=1;
				}
			}
	}
}

__global__ void
Kernel2( char* d_graph_mask, char *d_updatind_graph_mask, char* d_graph_visited, char *d_over, const int no_of_nodes)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if( tid<no_of_nodes && d_updatind_graph_mask[tid])
	{
		d_graph_mask[tid]=1;
		d_graph_visited[tid]=1;
		*d_over=1;
		d_updatind_graph_mask[tid]=0;
	}
}

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

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void run_bfs_gpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
    int *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
    char *h_graph_visited, int *h_cost) throw(std::string)
{

	Node* d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	int* d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;

	char* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(char)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice) ;

	char* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(char)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice) ;

	char* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(char)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice) ;

	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

  char h_over;
	char *d_over;
	cudaMalloc( (void**) &d_over, sizeof(char));

	// setup execution parameters
	dim3  grid( (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK);
	dim3  threads(MAX_THREADS_PER_BLOCK);

  do {
    h_over = 0;
		cudaMemcpy( d_over, &h_over, sizeof(char), cudaMemcpyHostToDevice) ;
		Kernel<<< grid, threads >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		Kernel2<<< grid, threads >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		cudaMemcpy( &h_over, d_over, sizeof(char), cudaMemcpyDeviceToHost) ;
	} while(h_over);


	// copy result from device to host
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) ;

	cudaFree(d_graph_nodes);
	cudaFree(d_graph_edges);
	cudaFree(d_graph_mask);
	cudaFree(d_updating_graph_mask);
	cudaFree(d_graph_visited);
	cudaFree(d_cost);

#ifdef	PROFILING
  kernel_timer.stop();
  kernel_time = kernel_timer.getTimeInSeconds();
  std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
}

//----------------------------------------------------------
//--cambine:	main function
//--author:		created by Jianbin Fang
//--date:		25/01/2011
//----------------------------------------------------------
int main(int argc, char * argv[])
{
  int no_of_nodes;
  int edge_list_size;
  FILE *fp;
  Node* h_graph_nodes;
  char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
  try{
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

    if(fp)
      fclose(fp);    
    // allocate mem for the result on host side
    int	*h_cost = (int*) malloc(sizeof(int)*no_of_nodes);
    int *h_cost_ref = (int*)malloc(sizeof(int)*no_of_nodes);
    for(int i=0;i<no_of_nodes;i++){
      h_cost[i]=-1;
      h_cost_ref[i] = -1;
    }
    h_cost[source]=0;
    h_cost_ref[source]=0;		
    //---------------------------------------------------------
    printf("run bfs (#nodes = %d) on device\n", no_of_nodes);
    run_bfs_gpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, 
                h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);	
    //---------------------------------------------------------
    //
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
    //---------------------------------------------------------
    //--result varification
    compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
    //release host memory		
    free(h_graph_nodes);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);

  }
  catch(std::string msg){
    std::cout<<"--cambine: exception in main ->"<<msg<<std::endl;
    //release host memory
    free(h_graph_nodes);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);		
  }

  return 0;
}
