#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>

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

void kernel(const unsigned int rows, const unsigned int cols,
		       const int cRows, const int contRows,
                       const unsigned char *snpdata,
		       float *results, sycl::nd_item<3> item_ct1)
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

        int tid = item_ct1.get_local_id(2) +
                  item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
        if (tid >= cols) return;
	
	cases[0]=1;cases[1]=1;cases[2]=1;
        controls[0]=1;controls[1]=1;controls[2]=1;
        
	// read cases: each thread reads a column of snpdata matrix
        for ( m = 0 ; m < cRows ; m++ ) {
	        y = snpdata[m * cols + tid];
                if ( y == '0') { cases[0]++; }
                else if ( y == '1') { cases[1]++; }
                else if ( y == '2') { cases[2]++; }
        }
               
	// read controls: each thread reads a column of snpdata matrix
	for ( n = cRows ; n < cRows + contRows ; n++ ) {
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
	results[tid] = chisquare;
}

int main(int argc, char *argv[]) try {

        /* Validation to check if the data file is readable */
	FILE *fp = fopen(argv[1], "r");
	if (fp == NULL) {
    		printf("Cannot Open the File");
		return 1;
	}

        /* snpmatrix size */
	size_t size;
        /* number of thread blocks */
	int BLOCKS;
	std::chrono::high_resolution_clock::time_point start, end;
	double seconds;
	unsigned int jobs; 
	unsigned long i;

	/*Kernel variable declaration */
	unsigned char *dev_dataT;
	float *results;
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

        /*
        DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        int err = (dpct::dev_mgr::instance().select_device(DEVICE), 0);

        dpct::device_info props;
        dpct::dev_mgr::instance().get_device(DEVICE).get_device_info(props);
        printf("Device name: %s\n", props.get_name());

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

	/* allocate the Memory in the GPU for SNP data */
        /*
        DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted.
        You may need to rewrite this code.
        */
        err = (dpct::dpct_malloc((void **)&dev_dataT,
                                 (size_t)size * (size_t)sizeof(unsigned char)),
               0);
        /*
        DPCT1000:1: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (err != 0) {
                printf("Error mallocing data on GPU device\n");
                /*
                DPCT1001:0: The statement could not be removed.
                */
                free(dataT);
                free(host_results);
		free(cpu_results);
		return 1;
	}
        /*
        DPCT1003:10: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        err = (dpct::dpct_malloc((void **)&results, cols * sizeof(float)), 0);
        /*
        DPCT1000:3: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (err != 0) {
                printf("Error mallocing results on GPU device\n");
                /*
                DPCT1001:2: The statement could not be removed.
                */
                free(dataT);
                free(host_results);
		free(cpu_results);
                dpct::dpct_free(dev_dataT);
                return 1;
	}

	/*Copy the SNP data to GPU */
        /*
        DPCT1003:11: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        err = (dpct::dpct_memcpy(dev_dataT, dataT,
                                 (size_t)size * (size_t)sizeof(unsigned char),
                                 dpct::host_to_device),
               0);
        /*
        DPCT1000:5: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (err != 0) {
                printf("Error copying data to GPU\n");
                /*
                DPCT1001:4: The statement could not be removed.
                */
                free(dataT);
                free(host_results);
		free(cpu_results);
                dpct::dpct_free(dev_dataT);
                dpct::dpct_free(results);
                return 1;
	}

	jobs = cols;
	BLOCKS = (jobs + THREADS - 1)/THREADS;

	/*Calling the kernel function */
        {
                dpct::buffer_t dev_dataT_buf_ct4 = dpct::get_buffer(dev_dataT);
                dpct::buffer_t results_buf_ct5 = dpct::get_buffer(results);
                dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                        auto dev_dataT_acc_ct4 =
                            dev_dataT_buf_ct4
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);
                        auto results_acc_ct5 =
                            results_buf_ct5
                                .get_access<sycl::access::mode::read_write>(
                                    cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS) *
                                                  sycl::range<3>(1, 1, THREADS),
                                              sycl::range<3>(1, 1, THREADS)),
                            [=](sycl::nd_item<3> item_ct1) {
                                    kernel(rows, cols, ncases, ncontrols,
                                           (const unsigned char
                                                *)(&dev_dataT_acc_ct4[0]),
                                           (float *)(&results_acc_ct5[0]),
                                           item_ct1);
                            });
                });
        }

        /*Copy the results back in host*/
        /*
        DPCT1003:12: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        err = (dpct::dpct_memcpy(host_results, results, cols * sizeof(float),
                                 dpct::device_to_host),
               0);
        /*
        DPCT1000:7: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (err != 0) {
                printf("Error copying data from GPU\n");
                /*
                DPCT1001:6: The statement could not be removed.
                */
                free(dataT);
                free(host_results);
		free(cpu_results);
                dpct::dpct_free(dev_dataT);
                dpct::dpct_free(results);
                return 1;
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
                        if (fabs(cpu_results[k] - host_results[k]) > 1e-4)
                            error++;
        }

        dpct::dpct_free(dev_dataT);
        dpct::dpct_free(results);
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
catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                  << ", line:" << __LINE__ << std::endl;
        std::exit(1);
}
