// thread block sizes
#define XXX 8
#define YYY 8

// data allocation on host and device
// data initialization on host
// host to device copy
# define TODEV(A,s) A = (float*) malloc ((s) * sizeof(float)); \
                    for (int i = 0; i < s; i++) A[i] = 0.001; \
                    float *A##_d = sycl::malloc_device<float>((s), Q); \
                    Q.memcpy(A##_d, A, sizeof(float) * (s));

// device to host copy
# define FROMDEV(A,s) Q.memcpy(A, A##_d, sizeof(float) * (s)).wait();

# define FREE(A) free(A); sycl::free(A##_d, Q);

# define TODEV3(A) TODEV(A,d3)
# define TODEV2(A) TODEV(A,d2)
# define FROMDEV3(A) FROMDEV(A,d3)
# define FROMDEV2(A) FROMDEV(A,d2)

