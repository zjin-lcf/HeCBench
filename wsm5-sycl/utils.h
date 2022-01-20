// thread block sizes
#define XXX 8
#define YYY 8

// data allocation on host and device
// data initialization on host
// host to device copy
# define TODEV(A,s) A = (float*) malloc ((s) * sizeof(float)); \
                    for (int i = 0; i < s; i++) A[i] = rand() / (float)RAND_MAX; \
                    buffer<float, 1> A##_d ((s)); \
                    Q.submit([&] (handler &cgh) {\
                      auto acc = A##_d.get_access<sycl_discard_write>(cgh); \
                      cgh.copy(A, acc);\
                    });

// device to host copy
# define FROMDEV(A,s) Q.submit([&] (handler &cgh) {\
                        auto acc = A##_d.get_access<sycl_read>(cgh); \
                        cgh.copy(acc, A);\
                      }).wait();

# define FREE(A) free(A)

# define TODEV3(A) TODEV(A,d3)
# define TODEV2(A) TODEV(A,d2)
# define FROMDEV3(A) FROMDEV(A,d3)
# define FROMDEV2(A) FROMDEV(A,d2)

