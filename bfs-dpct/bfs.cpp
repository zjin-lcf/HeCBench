#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

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

void
Kernel( const Node* d_graph_nodes, const int* d_graph_edges, char* d_graph_mask, char* d_updatind_graph_mask, const char *d_graph_visited, int* d_cost, const int no_of_nodes,
        sycl::nd_item<3> item_ct1) 
{
 int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
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

void
Kernel2( char* d_graph_mask, char *d_updatind_graph_mask, char* d_graph_visited, char *d_over, const int no_of_nodes,
         sycl::nd_item<3> item_ct1)
{
 int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
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
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();

        Node* d_graph_nodes;
 d_graph_nodes = sycl::malloc_device<Node>(no_of_nodes, q_ct1);
 q_ct1.memcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes).wait();

        int* d_graph_edges;
 d_graph_edges = sycl::malloc_device<int>(edge_list_size, q_ct1);
 q_ct1.memcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size)
     .wait();

        char* d_graph_mask;
 d_graph_mask = sycl::malloc_device<char>(no_of_nodes, q_ct1);
 q_ct1.memcpy(d_graph_mask, h_graph_mask, sizeof(char) * no_of_nodes).wait();

        char* d_updating_graph_mask;
 d_updating_graph_mask = sycl::malloc_device<char>(no_of_nodes, q_ct1);
 q_ct1
     .memcpy(d_updating_graph_mask, h_updating_graph_mask,
             sizeof(char) * no_of_nodes)
     .wait();

        char* d_graph_visited;
 d_graph_visited = sycl::malloc_device<char>(no_of_nodes, q_ct1);
 q_ct1.memcpy(d_graph_visited, h_graph_visited, sizeof(char) * no_of_nodes)
     .wait();

        int* d_cost;
 d_cost = sycl::malloc_device<int>(no_of_nodes, q_ct1);
 q_ct1.memcpy(d_cost, h_cost, sizeof(int) * no_of_nodes).wait();

  char h_over;
	char *d_over;
 d_over = sycl::malloc_device<char>(1, q_ct1);

        // setup execution parameters
 sycl::range<3> grid(
     (no_of_nodes + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, 1, 1);
 sycl::range<3> threads(MAX_THREADS_PER_BLOCK, 1, 1);

  do {
    h_over = 0;
  q_ct1.memcpy(d_over, &h_over, sizeof(char)).wait();
  q_ct1.submit([&](sycl::handler &cgh) {
   auto dpct_global_range = grid * threads;

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                          dpct_global_range.get(0)),
           sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        Kernel(d_graph_nodes, d_graph_edges, d_graph_mask,
               d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes,
               item_ct1);
       });
  });
  q_ct1.submit([&](sycl::handler &cgh) {
   auto dpct_global_range = grid * threads;

   cgh.parallel_for(
       sycl::nd_range<3>(
           sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                          dpct_global_range.get(0)),
           sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
       [=](sycl::nd_item<3> item_ct1) {
        Kernel2(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over,
                no_of_nodes, item_ct1);
       });
  });
  q_ct1.memcpy(&h_over, d_over, sizeof(char)).wait();
        } while(h_over);


	// copy result from device to host
 q_ct1.memcpy(h_cost, d_cost, sizeof(int) * no_of_nodes).wait();

 sycl::free(d_graph_nodes, q_ct1);
 sycl::free(d_graph_edges, q_ct1);
 sycl::free(d_graph_mask, q_ct1);
 sycl::free(d_updating_graph_mask, q_ct1);
 sycl::free(d_graph_visited, q_ct1);
 sycl::free(d_cost, q_ct1);

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
