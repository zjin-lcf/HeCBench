#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "common.h"

#define BLOCK_SIZE 16

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

static int do_verify = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

int main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;

  while ((opt = getopt_long(argc, argv, "::vs:i:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
        input_file = optarg;
        break;
      case 'v':
        do_verify = 1;
        break;
      case 's':
        matrix_dim = atoi(optarg);
        if (matrix_dim <= 0) {
          printf("Matrix dimension must be positive!\n");
          exit(EXIT_FAILURE);
        }
        if (matrix_dim % 16 != 0) {
          printf("Matrix dimension of %d not supported by the benchmark\n", matrix_dim);
          exit(EXIT_FAILURE);
        }
        printf("Generate input matrix internally, size =%d\n", matrix_dim);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
            argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }  

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } 

  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    // print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  /* beginning of timing point */
  stopwatch_start(&sw);

  #pragma omp target data map(tofrom: m[0:(size_t)matrix_dim*matrix_dim])
  {
  int offset;
  int i=0;
  
  auto start = std::chrono::steady_clock::now();

  for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
    offset = i;  // add the offset 
    #pragma omp target teams num_teams(1) thread_limit(BLOCK_SIZE)
    {
      float shadow[BLOCK_SIZE * BLOCK_SIZE];
      #pragma omp parallel
      {
        int i,j;
        int tx = omp_get_thread_num() ;
      
        size_t array_offset = offset*(size_t)matrix_dim+offset;
        for(i=0; i < BLOCK_SIZE; i++){
          shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
          array_offset += matrix_dim;
        }
        
        #pragma omp barrier
        
        for(i=0; i < BLOCK_SIZE-1; i++) {
      
          if (tx>i){
            for(j=0; j < i; j++)
              shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
          shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
          }
      
          #pragma omp barrier
          if (tx>i){
      
            for(j=0; j < i+1; j++)
              shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
          }
          
          #pragma omp barrier
        }
      
        array_offset = (offset+1)*(size_t)matrix_dim+offset;
        for(i=1; i < BLOCK_SIZE; i++){
          m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
          array_offset += matrix_dim;
        }
      }
    }

    #pragma omp target teams num_teams((matrix_dim-i)/BLOCK_SIZE-1) thread_limit(2*BLOCK_SIZE)
    {
      float dia[BLOCK_SIZE * BLOCK_SIZE];
      float peri_row[BLOCK_SIZE * BLOCK_SIZE];
      float peri_col[BLOCK_SIZE * BLOCK_SIZE];
      #pragma omp parallel
      {
         size_t array_offset;
         int i,j;
         int idx;

         int  bx = omp_get_team_num();  
         int  tx = omp_get_thread_num();

         if (tx < BLOCK_SIZE) {
           idx = tx;
           array_offset = offset*(size_t)matrix_dim+offset;
           for (i=0; i < BLOCK_SIZE/2; i++){
           dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
           array_offset += matrix_dim;
           }
         
         array_offset = offset*(size_t)matrix_dim+offset;
         for (i=0; i < BLOCK_SIZE; i++) {
           peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
           array_offset += matrix_dim;
         }

         } else {
         idx = tx-BLOCK_SIZE;
         
         array_offset = (offset+BLOCK_SIZE/2)*(size_t)matrix_dim+offset;
         for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
           dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
           array_offset += matrix_dim;
         }
         
         array_offset = (offset+(bx+1)*BLOCK_SIZE)*(size_t)matrix_dim+offset;
         for (i=0; i < BLOCK_SIZE; i++) {
           peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
           array_offset += matrix_dim;
         }
       }
       #pragma omp barrier

       if (tx < BLOCK_SIZE) { //peri-row
         idx=tx;
         for(i=1; i < BLOCK_SIZE; i++){
           for (j=0; j < i; j++)
             peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
         }
       } else { //peri-col
         idx=tx - BLOCK_SIZE;
         for(i=0; i < BLOCK_SIZE; i++){
           for(j=0; j < i; j++)
             peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
            peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
         }
       }

       #pragma omp barrier
       if (tx < BLOCK_SIZE) { //peri-row
         idx=tx;
         array_offset = (offset+1)*(size_t)matrix_dim+offset;
         for(i=1; i < BLOCK_SIZE; i++){
           m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
           array_offset += matrix_dim;
         }
       } else { //peri-col
         idx=tx - BLOCK_SIZE;
         array_offset = (offset+(bx+1)*BLOCK_SIZE)*(size_t)matrix_dim+offset;
         for(i=0; i < BLOCK_SIZE; i++){
           m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
           array_offset += matrix_dim;
         }
       }
      }
    }

    #pragma omp target teams num_teams(((matrix_dim-i)/BLOCK_SIZE-1) * ((matrix_dim-i)/BLOCK_SIZE-1)) \
                              thread_limit(BLOCK_SIZE*BLOCK_SIZE)
    {
      float peri_row[BLOCK_SIZE * BLOCK_SIZE];
      float peri_col[BLOCK_SIZE * BLOCK_SIZE];
      #pragma omp parallel
      {
        int  bx = omp_get_team_num() % ((matrix_dim-i)/BLOCK_SIZE-1); // item.get_group(1);  
        int  by = omp_get_team_num() / ((matrix_dim-i)/BLOCK_SIZE-1); //omp_get_team_num();  
        
        int  tx = omp_get_thread_num() % BLOCK_SIZE; //item.get_local_id(1);
        int  ty = omp_get_thread_num() / BLOCK_SIZE; //omp_get_thread_num();

        int i;
        float sum;

        int global_row_id = offset + (by+1)*BLOCK_SIZE;
        int global_col_id = offset + (bx+1)*BLOCK_SIZE;

        peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*(size_t)matrix_dim+global_col_id+tx];
        peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*(size_t)matrix_dim+offset+tx];

        #pragma omp barrier

        sum = 0;
        for (i=0; i < BLOCK_SIZE; i++)
          sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
        m[(global_row_id+ty)*(size_t)matrix_dim+global_col_id+tx] -= sum;
      }
    }
  } // for

  offset = i;  // add the offset 
  #pragma omp target teams num_teams(1) thread_limit(BLOCK_SIZE)
  {
    float shadow[BLOCK_SIZE * BLOCK_SIZE];
    #pragma omp parallel
    {
      int i,j;
      int tx = omp_get_thread_num() ;
    
      size_t array_offset = offset*(size_t)matrix_dim+offset;
      for(i=0; i < BLOCK_SIZE; i++){
        shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
        array_offset += matrix_dim;
      }
      
      #pragma omp barrier
      
      for(i=0; i < BLOCK_SIZE-1; i++) {
        if (tx>i) {
          for(j=0; j < i; j++)
            shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
          shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
        }
    
        #pragma omp barrier
        if (tx>i){
          for(j=0; j < i+1; j++)
            shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
        }

        #pragma omp barrier
      }
    
      array_offset = (offset+1)*(size_t)matrix_dim+offset;
      for(i=1; i < BLOCK_SIZE; i++){
        m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
        array_offset += matrix_dim;
      }
    }
   }

   auto end = std::chrono::steady_clock::now();
   auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
   printf("Total kernel execution time : %f (s)\n", time * 1e-9f);
  } // #pragma omp target  data map

  /* end of timing point */
  stopwatch_stop(&sw);
  printf("Device offloading time (s): %lf\n", get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    // print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);
}
