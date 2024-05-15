  miopenHandle_t h;
  checkMIOpen(miopenCreate(&h));

  miopenTensorDescriptor_t input_descriptor;
  checkMIOpen(miopenCreateTensorDescriptor(&input_descriptor));
  checkMIOpen(miopenSet4dTensorDescriptor(input_descriptor,
                                          /*dataType=*/miopenFloat,
                                          /*batch_size=*/N,
                                          /*channels=*/C,
                                          /*image_height=*/Hin,
                                          /*image_width=*/Win));

  miopenTensorDescriptor_t kernel_descriptor;
  checkMIOpen(miopenCreateTensorDescriptor(&kernel_descriptor));
  checkMIOpen(miopenSet4dTensorDescriptor(kernel_descriptor,
                                          /*dataType=*/miopenFloat,
                                          /*out_channels=*/M,
                                          /*in_channels=*/C,
                                          /*kernel_height=*/K,
                                          /*kernel_width=*/K));

  miopenConvolutionDescriptor_t convolution_descriptor;
  checkMIOpen(miopenCreateConvolutionDescriptor(&convolution_descriptor));
  checkMIOpen(miopenInitConvolutionDescriptor(convolution_descriptor,
                                              /*mode=*/miopenConvolution,
                                              /*pad_height=*/0,
                                              /*pad_width=*/0,
                                              /*vertical_stride=*/1,
                                              /*horizontal_stride=*/1,
                                              /*dilation_height=*/1,
                                              /*dilation_width=*/1));

  int batch_size{0}, channels{0}, height{0}, width{0};
  checkMIOpen(miopenGetConvolutionForwardOutputDim(convolution_descriptor,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   &batch_size,
                                                   &channels,
                                                   &height,
                                                   &width));

  #ifdef DEBUG
  std::cerr << "Output Image(NxHxWxC): " << batch_size << " x "
            << height << " x " << width << " x " << channels << std::endl;
  #endif

  miopenTensorDescriptor_t output_descriptor;
  checkMIOpen(miopenCreateTensorDescriptor(&output_descriptor));
  checkMIOpen(miopenSet4dTensorDescriptor(output_descriptor,
                                          /*dataType=*/miopenFloat,
                                          /*batch_size=*/N,
                                          /*channels=*/M,
                                          /*image_height=*/Hout,
                                          /*image_width=*/Wout));

  size_t workspace_bytes{0};
  checkMIOpen(miopenConvolutionForwardGetWorkSpaceSize(h,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       &workspace_bytes));
  void* d_workspace{nullptr};
  if (workspace_bytes > 0) {
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;
    hipMalloc(&d_workspace, workspace_bytes);
  }


  int requestedAlgoCount = 2;
  int returnedAlgoCount = -1;
  miopenConvAlgoPerf_t results[2];

  checkMIOpen(miopenFindConvolutionForwardAlgorithm(h,
                                                    input_descriptor,
                                                    dX, //d_input,
                                                    kernel_descriptor,
                                                    dW, //d_kernel,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    dY, //d_output
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    results,
                                                    d_workspace,
                                                    workspace_bytes,
                                                    true));

  #ifdef DEBUG
  std::cout << "Testing miopenFindConvolutionForwardAlgorithm ...\n";
  for(int algoIndex = 0; algoIndex < returnedAlgoCount; ++algoIndex){
    printf("^^^^ for Algo %d: %f time requiring %llu memory\n",
           results[algoIndex].fwd_algo, results[algoIndex].time,
           (unsigned long long)results[algoIndex].memory);
  }
  #endif

  miopenConvFwdAlgorithm_t convolution_algorithm = results[0].fwd_algo;

  const float alpha = 1.0f, beta = 0.0f;

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    checkMIOpen(miopenConvolutionForward(h,
                                         &alpha,
                                         input_descriptor,
                                         dX, //d_input,
                                         kernel_descriptor,
                                         dW, //d_kernel,
                                         convolution_descriptor,
                                         convolution_algorithm,
                                         &beta,
                                         output_descriptor,
                                         dY, //d_output
                                         d_workspace,
                                         workspace_bytes));
                                       
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s4 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  miopenDestroyTensorDescriptor(input_descriptor);
  miopenDestroyTensorDescriptor(output_descriptor);
  miopenDestroyTensorDescriptor(kernel_descriptor);
  miopenDestroyConvolutionDescriptor(convolution_descriptor);
  miopenDestroy(h);
