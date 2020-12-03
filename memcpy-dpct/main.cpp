#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define NUM_SIZE 16
#define NUM_ITER (1 << 13)

void setup(size_t *size) {
  for (int i = 0; i < NUM_SIZE; i++) {
    size[i] = 1 << (i + 6);  // start at 8 bytes
  }
}

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

int main() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *d_A;
  size_t size[NUM_SIZE];
  int err;

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    int* A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);

    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err = (d_A = (int *)sycl::malloc_device(size[i], q_ct1), 0);
    /*
    DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
      /*
      DPCT1001:0: The statement could not be removed.
      */
      std::cerr << "Device memory allocation failed\n";
      free(A);
      return -1;
    }

    clock_t start, end;
    double uS;

    /*
    DPCT1008:3: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      q_ct1.memcpy(d_A, A, size[i]);
    }
    dev_ct1.queues_wait_and_throw();
    /*
    DPCT1008:4: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    end = clock();
    uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;

    /*
    DPCT1008:5: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      q_ct1.memcpy(A, d_A, size[i]);
    }
    dev_ct1.queues_wait_and_throw();
    /*
    DPCT1008:6: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    end = clock();
    uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from device to host takes " 
      << uS <<  " us" << std::endl;

    sycl::free(d_A, q_ct1);
    free(A);
  }
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
