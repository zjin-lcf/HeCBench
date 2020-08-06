// Designed by: Amir Yazdanbakhsh
// Date: March 26th - 2015
// Alternative Computing Technologies Lab.
// Georgia Institute of Technology

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <fstream>

#define MAX_LOOP 25
#define MAX_DIFF 0.15f
#define NUM_JOINTS 3
#define PI 3.14159265358979f
#define NUM_JOINTS_P1 (NUM_JOINTS + 1)
#define BLOCK_SIZE 128

using namespace std;

void invkin_cpu(float *xTarget_in, float *yTarget_in, float *angles, int size)
{
	for (int idx = 0; idx < size; idx++) 
	{
		float angle_out[NUM_JOINTS];
		float xData[NUM_JOINTS_P1];
		float yData[NUM_JOINTS_P1];

		float curr_xTargetIn = xTarget_in[idx];
		float curr_yTargetIn = yTarget_in[idx];

		for(int i = 0; i < NUM_JOINTS; i++)
		{
			angle_out[i] = 0.0;
		}
		float angle;
		for (int i = 0 ; i < NUM_JOINTS_P1; i++)
		{
			xData[i] = i;
			yData[i] = 0.f;
		}

		for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
		{
			for (int iter = NUM_JOINTS; iter > 0; iter--) 
			{
				float pe_x = xData[NUM_JOINTS];
				float pe_y = yData[NUM_JOINTS];
				float pc_x = xData[iter-1];
				float pc_y = yData[iter-1];
				float diff_pe_pc_x = pe_x - pc_x;
				float diff_pe_pc_y = pe_y - pc_y;
				float diff_tgt_pc_x = curr_xTargetIn - pc_x;
				float diff_tgt_pc_y = curr_yTargetIn - pc_y;
    float len_diff_pe_pc =
        sqrtf(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
    float len_diff_tgt_pc =
        sqrtf(diff_tgt_pc_x * diff_tgt_pc_x + diff_tgt_pc_y * diff_tgt_pc_y);
                                float a_x = diff_pe_pc_x / len_diff_pe_pc;
				float a_y = diff_pe_pc_y / len_diff_pe_pc;
				float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
				float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
				float a_dot_b = a_x * b_x + a_y * b_y;
				if (a_dot_b > 1.f)
					a_dot_b = 1.f;
				else if (a_dot_b < -1.f)
					a_dot_b = -1.f;
    angle = acosf(a_dot_b) * (180.f / PI);
                                // Determine angle direction
				float direction = a_x * b_y - a_y * b_x;
				if (direction < 0.f)
					angle = -angle;
				// Make the result look more natural (these checks may be omitted)
				if (angle > 30.f)
					angle = 30.f;
				else if (angle < -30.f)
					angle = -30.f;
				// Save angle
				angle_out[iter - 1] = angle;
				for (int i = 0; i < NUM_JOINTS; i++) 
				{
					if(i < NUM_JOINTS - 1)
					{
						angle_out[i+1] += angle_out[i];
					}
				}
			}
		}

		angles[idx * NUM_JOINTS + 0] = angle_out[0];
		angles[idx * NUM_JOINTS + 1] = angle_out[1];
		angles[idx * NUM_JOINTS + 2] = angle_out[2];
	}

}

void invkin_kernel(float *xTarget_in, float *yTarget_in, float *angles, int size,
                   sycl::nd_item<3> item_ct1)
{

 int blockId = item_ct1.get_group(2) +
               item_ct1.get_group(1) * item_ct1.get_group_range(2);
 int idx = blockId * (item_ct1.get_local_range().get(2) *
                      item_ct1.get_local_range().get(1)) +
           (item_ct1.get_local_id(1) * item_ct1.get_local_range().get(2)) +
           item_ct1.get_local_id(2);

        if(idx < size)
	{	

		float angle_out[NUM_JOINTS];
		float curr_xTargetIn = xTarget_in[idx];
		float curr_yTargetIn = yTarget_in[idx];

		for(int i = 0; i < NUM_JOINTS; i++)
		{
			angle_out[i] = 0.0;
		}


		float angle;
		// Initialize x and y data
		float xData[NUM_JOINTS_P1];
		float yData[NUM_JOINTS_P1];

		for (int i = 0 ; i < NUM_JOINTS_P1; i++)
		{
			xData[i] = i;
			yData[i] = 0.f;
		}

		for(int curr_loop = 0; curr_loop < MAX_LOOP; curr_loop++)
		{
			for (int iter = NUM_JOINTS; iter > 0; iter--) 
			{
				float pe_x = xData[NUM_JOINTS];
				float pe_y = yData[NUM_JOINTS];
				float pc_x = xData[iter-1];
				float pc_y = yData[iter-1];
				float diff_pe_pc_x = pe_x - pc_x;
				float diff_pe_pc_y = pe_y - pc_y;
				float diff_tgt_pc_x = curr_xTargetIn - pc_x;
				float diff_tgt_pc_y = curr_yTargetIn - pc_y;
    float len_diff_pe_pc =
        sycl::sqrt(diff_pe_pc_x * diff_pe_pc_x + diff_pe_pc_y * diff_pe_pc_y);
    float len_diff_tgt_pc = sycl::sqrt(diff_tgt_pc_x * diff_tgt_pc_x +
                                       diff_tgt_pc_y * diff_tgt_pc_y);
                                float a_x = diff_pe_pc_x / len_diff_pe_pc;
				float a_y = diff_pe_pc_y / len_diff_pe_pc;
				float b_x = diff_tgt_pc_x / len_diff_tgt_pc;
				float b_y = diff_tgt_pc_y / len_diff_tgt_pc;
				float a_dot_b = a_x * b_x + a_y * b_y;
				if (a_dot_b > 1.f)
					a_dot_b = 1.f;
				else if (a_dot_b < -1.f)
					a_dot_b = -1.f;
    angle = sycl::acos(a_dot_b) * (180.f / PI);
                                // Determine angle direction
				float direction = a_x * b_y - a_y * b_x;
				if (direction < 0.f)
					angle = -angle;
				// Make the result look more natural (these checks may be omitted)
				if (angle > 30.f)
					angle = 30.f;
				else if (angle < -30.f)
					angle = -30.f;
				// Save angle
				angle_out[iter - 1] = angle;
				for (int i = 0; i < NUM_JOINTS; i++) 
				{
					if(i < NUM_JOINTS - 1)
					{
						angle_out[i+1] += angle_out[i];
					}
				}
			}
		}

		angles[idx * NUM_JOINTS + 0] = angle_out[0];
		angles[idx * NUM_JOINTS + 1] = angle_out[1];
		angles[idx * NUM_JOINTS + 2] = angle_out[2];
	}
}

int main(int argc, char *argv[]) try {
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
        if(argc != 3)
	{
		cerr << "Usage: ./invkin <input file coefficients>" << endl;
		exit(EXIT_FAILURE);
	}

	float* xTarget_in_h;
	float* yTarget_in_h;
	float* angle_out_h;
	float* angle_out_cpu;

 int cudaStatus;

        int data_size = 0;

	// process the files
	ifstream coordinate_in_file (argv[1]);
	const int iteration = atoi(argv[2]);


	if(coordinate_in_file.is_open())
	{
		coordinate_in_file >> data_size;
		cout << "# Data Size = " << data_size << endl;
	}

	// allocate the memory
	xTarget_in_h = new (nothrow) float[data_size];
	if(xTarget_in_h == NULL)
	{
		cerr << "Memory allocation fails!!!" << endl;
		exit(EXIT_FAILURE);	
	}
	yTarget_in_h = new (nothrow) float[data_size];
	if(yTarget_in_h == NULL)
	{
		cerr << "Memory allocation fails!!!" << endl;
		exit(EXIT_FAILURE);	
	}
	angle_out_h = new (nothrow) float[data_size*NUM_JOINTS];
	if(angle_out_h == NULL)
	{
		cerr << "Memory allocation fails!!!" << endl;
		exit(EXIT_FAILURE);	
	}

	angle_out_cpu = new (nothrow) float[data_size*NUM_JOINTS];
	if(angle_out_cpu == NULL)
	{
		cerr << "Memory allocation fails!!!" << endl;
		exit(EXIT_FAILURE);	
	}



	// Prepare
 sycl::event start, stop;
 /*
 DPCT1026:0: The call to cudaEventCreate was removed, because this call is
 redundant in DPC++.
 */
 /*
 DPCT1026:1: The call to cudaEventCreate was removed, because this call is
 redundant in DPC++.
 */

        // add data to the arrays
	float xTarget_tmp, yTarget_tmp;
	int coeff_index = 0;
	while(coeff_index < data_size)
	{	
		coordinate_in_file >> xTarget_tmp >> yTarget_tmp;

		for(int i = 0; i < NUM_JOINTS ; i++)
		{
			angle_out_h[coeff_index * NUM_JOINTS + i] = 0.0;
		}

		xTarget_in_h[coeff_index] = xTarget_tmp;
		yTarget_in_h[coeff_index++] = yTarget_tmp;
	}


	cout << "# Coordinates are read from file..." << endl;

	// memory allocations on the host
	float 	*xTarget_in_d,
		*yTarget_in_d;
	float 	*angle_out_d;

 xTarget_in_d = sycl::malloc_device<float>(data_size, q_ct1);
 yTarget_in_d = sycl::malloc_device<float>(data_size, q_ct1);
 angle_out_d = sycl::malloc_device<float>(data_size * NUM_JOINTS, q_ct1);

        cout << "# Memory allocation on GPU is done..." << endl;

 q_ct1.memcpy(xTarget_in_d, xTarget_in_h, data_size * sizeof(float)).wait();
 q_ct1.memcpy(yTarget_in_d, yTarget_in_h, data_size * sizeof(float)).wait();

        cout << "# Data are transfered to GPU..." << endl;

 sycl::range<3> dimBlock(BLOCK_SIZE, 1, 1);
 sycl::range<3> dimGrid((data_size + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);


 for (int n = 0; n < iteration; n++)
  q_ct1.submit([&](sycl::handler &cgh) {
  auto dpct_global_range = dimGrid * dimBlock;

  cgh.parallel_for(
      sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                         dpct_global_range.get(0)),
          sycl::range<3>(dimBlock.get(2), dimBlock.get(1), dimBlock.get(0))),
      [=](sycl::nd_item<3> item_ct1) {
       invkin_kernel(xTarget_in_d, yTarget_in_d, angle_out_d, data_size,
                     item_ct1);
      });
 });
 /*
 DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
 may need to rewrite this code.
 */
 cudaStatus = (dev_ct1.queues_wait_and_throw(), 0);
 /*
 DPCT1000:4: Error handling if-stmt was detected but could not be rewritten.
 */
 if (cudaStatus != 0) {
  /*
  DPCT1001:3: The statement could not be removed.
  */
  cout << "Something was wrong! Error code: " << cudaStatus << endl;
        }


 q_ct1.memcpy(angle_out_h, angle_out_d, data_size * NUM_JOINTS * sizeof(float))
     .wait();

        // CPU
	invkin_cpu(xTarget_in_h, yTarget_in_h, angle_out_cpu, data_size);

	// Check
	int error = 0;
	for(int i = 0; i < data_size; i++)
	{
		for(int j = 0 ; j < NUM_JOINTS; j++)
		{
   if (fabsf(angle_out_h[i * NUM_JOINTS + j] -
             angle_out_cpu[i * NUM_JOINTS + j]) > 1e-3)
                                        error++;
		} 
	}

	// close files
	coordinate_in_file.close();

	// de-allocate the memory
	delete[] xTarget_in_h;
	delete[] yTarget_in_h;
	delete[] angle_out_h;
	delete[] angle_out_cpu;

	// de-allocate cuda memory
 sycl::free(xTarget_in_d, q_ct1);
 sycl::free(yTarget_in_d, q_ct1);
 sycl::free(angle_out_d, q_ct1);

        if (error) 
	{
		cout << "FAILED\n";
		return 1;
	} 
	else 
	{
		cout << "PASSED\n";
		return 0;
	}
}
catch (sycl::exception const &exc) {
 std::cerr << exc.what() << "Exception caught at file:" << __FILE__
           << ", line:" << __LINE__ << std::endl;
 std::exit(1);
}
