#include <iostream>
#include <string>
#include <chrono>

#include "bm3d.hpp"
#define cimg_display 0
#include "CImg.h"

// Repeat the execution of kernels 100 times
#define REPEAT 100

// Adjust the size of the total shared local memory for different GPUs
// e.g. 48KB on P100
#define TOTAL_SLM     48*1024

// Adjust the thread block size of the block matching kernel for different GPUs. 
// The maximum thread block size is 32 * MAX_NUM_WARPS
#define MAX_NUM_WARPS 16u

using namespace cimg_library;

int main(int argc, char** argv)
{
  if( argc < 4 )
  {
    std::cerr << "Usage: " << argv[0]
      << " NosiyImage DenoisedImage sigma [color] [ReferenceImage]\n"
      << "   color - color image denoising (experimental only)\n"
      << "   ReferenceImage - if provided, computes and prints PSNR " 
      << "between the reference image and denoised image\n";
    return 1;
  }

  //Store a noisy image
  CImg<unsigned char> image(argv[1]);

  float sigma = strtof(argv[3], NULL);

  unsigned int channels = 1;
  if (argc >= 5 && strcmp(argv[4],"color") == 0) channels = 3;

  std::cout << "Sigma = " << sigma << std::endl;

  if (channels > 1)
    std::cout << "Color denoising: yes" << std::endl;
  else
    std::cout << "Color denoising: no" << std::endl;

  std::vector<unsigned int> sigma2(channels);
  sigma2[0] = 25 * 25;

  //Convert color image to YCbCr color space
  if (channels == 3)
  {
    image = image.get_channels(0, 2).RGBtoYCbCr();
    //Convert the sigma^2 variance to the YCbCr color space
    long s = sigma * sigma;
    sigma2[0] = ((66l*66l*s + 129l*129l*s + 25l*25l*s) / (256l*256l));
    sigma2[1] = ((38l*38l*s + 74l*74l*s + 112l*112l*s) / (256l*256l));
    sigma2[2] = ((112l*112l*s + 94l*94l*s + 18l*18l*s) / (256l*256l));
  }

  std::cout << "Noise variance for individual channels (YCrCb if color): ";
  for (unsigned int k = 0; k < sigma2.size(); k++)
    std::cout << sigma2[k] << " ";
  std::cout << std::endl;

  // Check for invalid input
  if(! image.data() )              
  {
    std::cerr << "Could not open or find the image" << std::endl;
    return 1;
  }

  std::cout << "Image width: " << image.width() << " height: " << image.height() << std::endl;

  //Store a denoised image
  CImg<unsigned char> dst_image(image.width(), image.height(), 1, channels, 0);

  // Vector of image channels
  std::vector<sycl::uchar*> d_noisy_image;
  std::vector<sycl::uchar*> d_denoised_image;
  //Numerator and denominator used for aggregation
  std::vector<float*> d_numerator;  
  std::vector<float*> d_denominator;

  sycl::uint2 h_batch_size (256, 128);         //h_batch_size.x() has to be divisible by properties.warpSize

  //Denoising parameters and their shorthands
  Params h_hard_params(19, 8, 16, 2500, 3, 2.7f);
  const uint k = h_hard_params.k;
  const uint N = h_hard_params.N;
  const uint p = h_hard_params.p;

  //Reserved sizes
  const int width = image.width();
  const int height = image.height();
  size_t image_size = width * height;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  d_noisy_image.resize(channels);
  d_denoised_image.resize(channels);
  d_numerator.resize(channels);
  d_denominator.resize(channels);

  for(auto & it : d_noisy_image) {
    it = sycl::malloc_device<sycl::uchar>(image_size, q);
  }
  
  for(auto & it : d_denoised_image)
    it = sycl::malloc_device<sycl::uchar>(image_size, q);

  for(auto & it : d_numerator) 
    it = sycl::malloc_device<float>(image_size, q);

  for(auto & it : d_denominator)
    it = sycl::malloc_device<float>(image_size, q);

  const int batch_size = h_batch_size.x() * h_batch_size.y();

  //Addresses of similar patches to each reference patch of a batch
  ushort *d_stacks = sycl::malloc_device<ushort>(batch_size * N, q); 

  //Number of similar patches for each referenca patch of a batch that are stored in d_stacks
  uint *d_num_patches_in_stack = sycl::malloc_device<uint>(batch_size, q);

  //3D groups of a batch
  float *d_gathered_stacks = sycl::malloc_device<float>((N+1) * k * k * batch_size, q);

  //Weights for aggregation
  float *d_w_P = sycl::malloc_device<float>(batch_size, q);

  //image dimensions
  const sycl::uint2 image_dim (width, height);

  //dimensions limiting addresses of reference patches
  const sycl::uint2 stacks_dim (width - (k - 1), height - (k - 1));

  int paramN1 = N + 1; //maximal size of a stack with a reference patch

  const uint p_block_width = (warpSize-1) * p + k;
  const uint s_image_p_size = p_block_width * k * sizeof(sycl::uchar);

  const uint shared_mem_available = TOTAL_SLM - s_image_p_size;

  //Block-matching shared memory sizes per warp
  const uint s_diff_size = p_block_width * sizeof(uint);
  const uint s_patches_in_stack_size = warpSize * sizeof(sycl::uchar);
  const uint s_patch_stacks_size = N * warpSize * sizeof(uint);

  const uint num_warps = std::min(shared_mem_available / 
    (s_diff_size + s_patches_in_stack_size + s_patch_stacks_size), MAX_NUM_WARPS);
  uint lmem_size_bm = ((s_diff_size + s_patches_in_stack_size + s_patch_stacks_size) * num_warps) + 
    s_image_p_size;    

  //Determine launch parameteres for the block match kernel
  const sycl::range<2> lws_bm (1, warpSize*num_warps);
  const sycl::range<2> gws_bm (h_batch_size.y(), h_batch_size.x() * num_warps);

  //Determine launch parameteres for the get and aggregate kernels
  const sycl::range<2> lws (k, k);
  const sycl::range<2> gws (h_batch_size.y() * k, h_batch_size.x() * k);

  //Determine launch parameteres for the DCT kernel
  const uint trans_size = k*k*paramN1*batch_size;
  const sycl::range<2> gws_tr (KER2_BLOCK_WIDTH/k, (trans_size + (KER2_BLOCK_WIDTH*k) - 1) / (KER2_BLOCK_WIDTH*k) * k);
  const sycl::range<2> lws_tr (KER2_BLOCK_WIDTH/k, k);

  const uint s_size_t = k*k*(paramN1+1)*sizeof(float); //+1 for avoinding bank conflicts

  //Determine launch parameteres for final division kernel
  const sycl::range<2> lws_f(4, 64);
  const sycl::range<2> gws_f((height + 3)/4*4, (width + 63)/64*64);

  //Create an kaiser window (only for k = 8, alpha = 2.0) and copy it to the device.
  std::vector<float> kaiserWindow(k*k);
  if (k == 8) {
    // First quarter of the matrix
    kaiserWindow[0 + k * 0] = 0.1924f; 
    kaiserWindow[0 + k * 1] = 0.2989f;
    kaiserWindow[0 + k * 2] = 0.3846f;
    kaiserWindow[0 + k * 3] = 0.4325f;
    kaiserWindow[1 + k * 0] = 0.2989f;
    kaiserWindow[1 + k * 1] = 0.4642f;
    kaiserWindow[1 + k * 2] = 0.5974f;
    kaiserWindow[1 + k * 3] = 0.6717f;
    kaiserWindow[2 + k * 0] = 0.3846f;
    kaiserWindow[2 + k * 1] = 0.5974f;
    kaiserWindow[2 + k * 2] = 0.7688f;
    kaiserWindow[2 + k * 3] = 0.8644f;
    kaiserWindow[3 + k * 0] = 0.4325f;
    kaiserWindow[3 + k * 1] = 0.6717f;
    kaiserWindow[3 + k * 2] = 0.8644f; 
    kaiserWindow[3 + k * 3] = 0.9718f;

    // Fill the rest of the matrix by symmetry
    for(unsigned i = 0; i < k / 2; i++)
      for (unsigned j = k / 2; j < k; j++)
        kaiserWindow[i + k * j] = kaiserWindow[i + k * (k - j - 1)];

    for (unsigned i = k / 2; i < k; i++)
      for (unsigned j = 0; j < k; j++)
        kaiserWindow[i + k * j] = kaiserWindow[k - i - 1 + k * j];
  }
  else
    for (unsigned i = 0; i < k * k; i++)
      kaiserWindow[i] = 1.0f;

  //Kaiser window used for aggregation
  float *d_kaiser_window = sycl::malloc_device<float>(k * k, q);
  q.memcpy(d_kaiser_window, kaiserWindow.data(), k * k * sizeof(float));

  // Copy images to device
  for(uint i = 0; i < channels; ++i) 
    q.memcpy(d_noisy_image[i], image.data()+i*image_size, image_size * sizeof(sycl::uchar));

  q.wait();

  // start measuring the total time
  auto start = std::chrono::high_resolution_clock::now();

  // repeat the execution of kernels
  for (int n = 0; n < REPEAT; n++) {

    for(auto & it : d_numerator) 
      q.memset(it, 0, image_size * sizeof(float));

    for(auto & it : d_denominator)
      q.memset(it, 0, image_size * sizeof(float));

    //Batch processing: in each iteration only the batch_size reference patches are processed. 
    sycl::uint2 start_point;
    for(start_point.y() = 0; start_point.y() < stacks_dim.y() + p - 1; 
        start_point.y() += (h_batch_size.y()*p))
    {
      for(start_point.x() = 0; start_point.x() < stacks_dim.x() + p - 1; 
          start_point.x() += (h_batch_size.x()*p))
      {
        //Finds similar patches for each reference patch of a batch and stores them in d_stacks array
        run_block_matching(
            q,
            d_noisy_image[0],      // IN: Image  
            d_stacks,              // OUT: Array of adresses of similar patches
            d_num_patches_in_stack,// OUT: Array containing numbers of these addresses
            image_dim,             // IN: Image dimensions
            stacks_dim,            // IN: Dimensions limiting addresses of reference patches
            h_hard_params,         // IN: Denoising parameters 
            start_point,           // IN: Address of the top-left reference patch of a batch
            lws_bm,                // Local work size
            gws_bm,                // Global work size
            lmem_size_bm           // Shared memory size in bytes
        );

        for (uint channel = 0; channel < channels; ++channel)
        {
          //Assembles 3D groups of a batch according to the d_stacks array
          run_get_block(
              q,
              start_point,             // IN: First reference patch of a batch
              d_noisy_image[channel],  // IN: Image
              d_stacks,                // IN: Array of adresses of similar patches
              d_num_patches_in_stack,  // IN: Numbers of patches in 3D groups
              d_gathered_stacks,       // OUT: Assembled 3D groups
              image_dim,               // IN: Image dimensions
              stacks_dim,              // IN: Dimensions limiting addresses of reference patches
              h_hard_params,           // IN: Denoising parameters
              lws,                     // Local work size
              gws                      // Global work size
          );

          //Apply the 2D DCT transform to each layer of 3D group
          run_DCT2D8x8(q, d_gathered_stacks, d_gathered_stacks, trans_size, 
                       lws_tr, gws_tr);

          // 1) 1D Walsh-Hadamard transform of proper size on the 3rd dimension of each 
          //      3D group of a batch to complete the 3D transform.
          // 2) Hard thresholding
          // 3) Inverse 1D Walsh-Hadamard trannsform.
          // 4) Compute the weingt of each 3D group

          run_hard_treshold_block(
              q,
              start_point,           // IN: First reference patch of a batch
              d_gathered_stacks,     // IN/OUT: 3D groups with transfomed patches
              d_w_P,                 // OUT: Weight of each 3D group
              d_num_patches_in_stack,// IN: Numbers of patches in 3D groups
              stacks_dim,            // IN: Dimensions limiting addresses of reference patches
              h_hard_params,         // IN: Denoising parameters
              sigma2[channel],       // IN: sigma
              lws,                   // Threads in block
              gws,                   // Blocks in grid
              s_size_t               // Shared memory size
          );

          //Apply inverse 2D DCT transform to each layer of 3D group
          run_IDCT2D8x8(q, d_gathered_stacks, d_gathered_stacks, trans_size, lws_tr, gws_tr);

          //Aggregates filtered patches of all 3D groups of a batch into numerator and denominator buffers
          run_aggregate_block(
              q,
              start_point,           // IN: First reference patch of a batch
              d_gathered_stacks,     // IN: 3D groups with transfomed patches
              d_w_P,                 // IN: Numbers of non zero coeficients after 3D thresholding
              d_stacks,              // IN: Array of adresses of similar patches
              d_kaiser_window,       // IN: Kaiser window
              d_numerator[channel],  // IN/OUT: Numerator aggregation buffer
              d_denominator[channel],// IN/OUT: Denominator aggregation buffer
              d_num_patches_in_stack,// IN: Numbers of patches in 3D groups
              image_dim,             // IN: Image dimensions
              stacks_dim,            // IN: Dimensions limiting addresses of reference patches
              h_hard_params,         // IN: Denoising parameters
              lws,                   // Threads in block
              gws                    // Blocks in grid
          );
        }
      }
    }

    //Divide numerator by denominator and save the result in output image
    for (uint channel = 0; channel < channels; ++channel)
    {
      run_aggregate_final(
          q,
          d_numerator[channel],      // IN: Numerator aggregation buffer
          d_denominator[channel],    // IN: Denominator aggregation buffer
          image_dim,                 // IN: Image dimensions
          d_denoised_image[channel], // OUT: Image estimate
          lws_f,                     // Threads in block
          gws_f                      // Blocks in grid
      );
    }
  } // REPEAT

  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double gpuTime = (double)elapsed_seconds.count();
  std::cout << "Average device execution time (s): " << gpuTime / REPEAT << std::endl;

  for (uint channel = 0; channel < channels; ++channel) {
    q.memcpy(dst_image.data()+channel*image_size,
             d_denoised_image[channel],
            image_size*sizeof(sycl::uchar)).wait();
  }

  if (channels == 3) 
    dst_image = dst_image.get_channels(0,2).YCbCrtoRGB();
  else
    dst_image = dst_image.get_channel(0);

  //Save denoised image
  dst_image.save( argv[2] );

  if (argc >= 6) {
    CImg<unsigned char> reference_image(argv[5]);
    std::cout << "PSNR:" << reference_image.PSNR(dst_image) << std::endl;
  }

  sycl::free(d_stacks, q);
  sycl::free(d_num_patches_in_stack, q);
  sycl::free(d_gathered_stacks, q);
  sycl::free(d_w_P, q);
  sycl::free(d_kaiser_window, q);

  for (auto & it : d_noisy_image)
    sycl::free(it, q);
  d_noisy_image.clear();

  for (auto & it : d_denoised_image)
    sycl::free(it, q);
  d_denoised_image.clear();

  for(auto & it : d_numerator)
    sycl::free(it, q);
  d_numerator.clear();

  for(auto & it : d_denominator)
    sycl::free(it, q);
  d_denominator.clear();

  return 0;
}
