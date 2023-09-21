// thread block sizes
#define XXX 8
#define YYY 8

// data allocation on host and device
// data initialization on host
# define ALLOC(A,s) A = (float*) malloc ((s) * sizeof(float)); \
                    for (int i = 0; i < s; i++) A[i] = 0.001;

// deallocation host and device memory
# define FREE(A) free(A);

# define ALLOC3(A) ALLOC(A,d3)
# define ALLOC2(A) ALLOC(A,d2)

