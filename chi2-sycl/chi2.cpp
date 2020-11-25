#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include "common.h"

void cpu_kernel(const unsigned int rows, 
		const unsigned int cols , 
		const int cRows , 
		const int contRows ,
		const unsigned char *snpdata,
		float *results)
{
        unsigned char y;
        int m, n ;
        unsigned int p = 0 ;
        int cases[3];
        int controls[3];
        int tot_cases = 1;
        int tot_controls= 1;
        int total = 1;
        float chisquare = 0.0f;
        float exp[3];
        float Conexpected[3];
        float Cexpected[3];
        float numerator1;
        float numerator2;
	unsigned int tid=0;

 	for(tid = 0; tid < cols; tid++) {
		
		chisquare=0;
	        cases[0]=1;cases[1]=1;cases[2]=1;
        	controls[0]=1;controls[1]=1;controls[2]=1;

	        for ( m = 0 ; m < cRows ; m++ ) {
        	        y = snpdata[(size_t) ((size_t) m) * ((size_t) cols) + tid];
                	if ( y == '0') { cases[0]++; }
	                else if ( y == '1') { cases[1]++; }
        	        else if ( y == '2') { cases[2]++; }
	        }

		for ( n = cRows ; n < cRows + contRows ; n++ ) {
	                y = snpdata[(size_t) ((size_t) n) * ((size_t) cols) + tid];
        	        if ( y == '0' ) { controls[0]++; }
                	else if ( y == '1') { controls[1]++; }
	                else if ( y == '2') { controls[2]++; }
        	}

		tot_cases = cases[0]+cases[1]+cases[2];
        	tot_controls = controls[0]+controls[1]+controls[2];
	        total = tot_cases + tot_controls;

	        for( p = 0 ; p < 3; p++) {
        	        exp[p] = (float)cases[p] + controls[p];
                	Cexpected[p] = tot_cases * exp[p] / total;
	                Conexpected[p] = tot_controls * exp[p] / total;
        	        numerator1 = (float)cases[p] - Cexpected[p];
                	numerator2 = (float)controls[p] - Conexpected[p];
	                chisquare += numerator1 * numerator1 / Cexpected[p] +  numerator2 * numerator2 / Conexpected[p];
        	}

	        results[tid] = chisquare;
	}
}


int main(int argc ,char* argv[]) {

	/* Validation to check if the data file is readable */
	FILE *fp = fopen(argv[1], "r");
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 1;
	}

        /* snpmatrix size */
	size_t size;
        /* number of thread blocks */
	std::chrono::high_resolution_clock::time_point start, end;
	double seconds;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
        char *line = NULL; size_t len = 0;
	char *token, *saveptr;
	  
	/* Initialize rows, cols, ncases, ncontrols from the user */
	unsigned int rows=atoi(argv[2]);
	unsigned int cols=atoi(argv[3]);
	int ncases=atoi(argv[4]);
	int ncontrols=atoi(argv[5]);
	int DEVICE = atoi(argv[6]);
	int THREADS = atoi(argv[7]);
	printf("Individuals=%d SNPs=%d cases=%d controls=%d DEVICE=%d THREADS=%d\n",
			rows,cols,ncases,ncontrols,DEVICE,THREADS);

	size = (size_t)((size_t)rows * (size_t)cols);
	printf("Size of the data = %lu\n",size);

	fflush(stdout);

	unsigned char *dataT = (unsigned char*)malloc((size_t)size);
	if(dataT == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
		return 1;
	}

	float* host_results = (float*) malloc(cols * sizeof(float)); 
	if(host_results == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
		free(dataT);
		return 1;
	}

	float* cpu_results = (float*) malloc(cols * sizeof(float)); 
	if(cpu_results == NULL) {
	        printf("ERROR: Memory for data not allocated.\n");
		free(dataT);
		free(host_results);
		return 1;
	}

        /* Transfer the SNP Data from the file to CPU Memory */
        i=0;
	while (getline(&line, &len, fp) != -1) {
                token = strtok_r(line, " ", &saveptr);
                while(token != NULL){
                        dataT[i] = *token;
                        i++;
                        token = strtok_r(NULL, " ", &saveptr);
                }
  	}
	fclose(fp);
        printf("Finished read the SNP data from the file.\n");
        fflush(stdout);


	// Get device time
	start = std::chrono::high_resolution_clock::now();

        { 
      #ifdef USE_GPU
        gpu_selector dev_sel;
      #else
        cpu_selector dev_sel;
      #endif
        queue q(dev_sel);

	/* allocate the Memory in the GPU for SNP data */	   
        const property_list props = property::buffer::use_host_ptr();
	buffer<unsigned char, 1> dev_dataT (dataT, size, props);
	buffer<float, 1> results (host_results, cols, props);

	jobs = cols;
	size_t global_work_size = (jobs + THREADS - 1)/THREADS * THREADS;

	q.submit([&](handler& cgh) {
          auto snpdata = dev_dataT.get_access<sycl_read>(cgh);
          auto chi_result = results.get_access<sycl_write>(cgh);
          cgh.parallel_for<class chi2>(
            nd_range<1>(range<1>(global_work_size), range<1>(THREADS)), [=] (nd_item<1> item) {
              unsigned char y;
              int m, n ;
              unsigned int p = 0 ;
              int cases[3];
              int controls[3];
              int tot_cases = 1;
              int tot_controls= 1;
              int total = 1;
              float chisquare = 0.0f;
              float exp[3];        
              float Conexpected[3];        
              float Cexpected[3];
              float numerator1;
              float numerator2;
              
              int tid  = item.get_global_id(0);
              if (tid >= cols) return;
              
              cases[0]=1;cases[1]=1;cases[2]=1;
              controls[0]=1;controls[1]=1;controls[2]=1;
              
              // read cases: each thread reads a column of snpdata matrix
              for ( m = 0 ; m < ncases ; m++ ) {
              	y = snpdata[m * cols + tid];
              	if ( y == '0') { cases[0]++; }
              	else if ( y == '1') { cases[1]++; }
              	else if ( y == '2') { cases[2]++; }
              }
              
              // read controls: each thread reads a column of snpdata matrix
              for ( n = ncases ; n < ncases + ncontrols ; n++ ) {
              	y = snpdata[n * cols + tid];
              	if ( y == '0' ) { controls[0]++; }
              	else if ( y == '1') { controls[1]++; }
              	else if ( y == '2') { controls[2]++; }
              }
              
              tot_cases = cases[0]+cases[1]+cases[2];
              tot_controls = controls[0]+controls[1]+controls[2];
              total = tot_cases + tot_controls;
              
              for( p = 0 ; p < 3; p++) {
              	exp[p] = (float)cases[p] + controls[p]; 
              	Cexpected[p] = tot_cases * exp[p] / total;
              	Conexpected[p] = tot_controls * exp[p] / total;
              	numerator1 = (float)cases[p] - Cexpected[p];
              	numerator2 = (float)controls[p] - Conexpected[p];
              	chisquare += numerator1 * numerator1 / Cexpected[p] +  numerator2 * numerator2 / Conexpected[p];
              }
              chi_result[tid] = chisquare;
           });
        });
	}
	end = std::chrono::high_resolution_clock::now();
	seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	printf("Total time (device) = %f\n", seconds);

	/* Get cpu time of serial execution */
	start = std::chrono::high_resolution_clock::now();

	cpu_kernel(rows,cols,ncases,ncontrols,dataT,cpu_results);

	end = std::chrono::high_resolution_clock::now();
	seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
	printf("Total time (cpu) = %f\n", seconds);

	/* verify using the cpu results */
	int error = 0;
	for(unsigned int k = 0; k < jobs; k++) {
		if (std::fabs(cpu_results[k] - host_results[k]) > 1e-4) error++;
	}

	free(dataT);
	free(host_results);
	free(cpu_results);

	if (error) {
	  printf("FAILED\n");
	  return EXIT_FAILURE;
	} else {
	  printf("PASSED\n");
	  return EXIT_SUCCESS;
	}

}
