#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstring>
#include <list>
#include <iterator>
#include <mkl_blas_sycl.hpp>
#include <mkl_lapack_sycl.hpp>
#include <mkl_sycl_types.hpp>
#include <dpct/blas_utils.hpp>

// mkl/sycl includes

template <typename T>
void print_2x2_matrix_values(T M, int ldM, std::string M_name)
{

    std::cout << std::endl;
    std::cout << "\t\t\t" << M_name << " = [ " << M[0*ldM + 0] << ", " << M[1*ldM + 0]         << ", ...\n";
    std::cout << "\t\t\t    [ "                << M[0*ldM + 1] << ", " << M[1*ldM + 1] << ", ...\n";
    std::cout << "\t\t\t    [ "                << "...\n";
    std::cout << std::endl;
}

//
// helpers for initializing templated scalar data type values.
//
template <typename fp> void rand_matrix(fp *M, char trans, int m, int n, int ld)
{

    if (trans == 'N') {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                M[i + j * ld] = rand() % 5;
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                M[j + i * ld] = rand() % 5;
    }
}

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp> void run_gemm_example() {
        dpct::device_ext &dev_ct1 = dpct::get_current_device();
        sycl::queue &q_ct1 = dev_ct1.default_queue();

    //
    // Initialize data for Gemm
    //
    // C = alpha * op(A) * op(B)  + beta * C
    //

    // matrix data sizes
    int m = 79;
    int n = 83;
    int k = 91;

    // leading dimensions of data
    int ldA = m;
    int ldB = k;
    int ldC = m;

    // set scalar fp values
    const fp alpha = fp(2.0);
    const fp beta  = fp(0.5);

    // prepare matrix data
    fp* a = (float *) aligned_alloc(64, (m * k) * sizeof(float));
    fp* b = (float *) aligned_alloc(64, (k * n) * sizeof(float));
    fp* c = (float *) aligned_alloc(64, (m * n) * sizeof(float));

    rand_matrix(a, 'T', m, k, ldA);
    rand_matrix(b, 'N', k, n, ldB);
    rand_matrix(c, 'N', m, n, ldC);

    {
    float *da, *db, *dc;
                da = sycl::malloc_device<float>((m * k), q_ct1);
                db = sycl::malloc_device<float>((k * n), q_ct1);
                dc = sycl::malloc_device<float>((m * n), q_ct1);
                q_ct1.memcpy(da, a, (m * k) * sizeof(float)).wait();
                q_ct1.memcpy(db, b, (k * n) * sizeof(float)).wait();
                q_ct1.memcpy(dc, c, (m * n) * sizeof(float)).wait();

    // create execution queue and buffers of matrix data
                sycl::queue *h;
                h = &q_ct1;

		for (int i = 0; i < 20000; i++)
                mkl::blas::gemm(*h, mkl::transpose::trans,
                                mkl::transpose::nontrans, m, n, k,
                                dpct::get_value(&alpha, *h), da, ldA, db, ldB,
                                dpct::get_value(&beta, *h), dc, ldC);

                q_ct1.memcpy(c, dc, (m * n) * sizeof(float)).wait();
                h = nullptr;

                sycl::free(da, q_ct1);
                sycl::free(db, q_ct1);
                sycl::free(dc, q_ct1);
    }

    //
    // Post Processing
    //

    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(a, ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(b, ldB, "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(c, ldC, "C");

    free(a);
    free(b);
    free(c);

}


//
// Main entry point for example.
//
int main (int argc, char ** argv) {
        srand(2);
        std::cout << "\tRunning with single precision real data type:" << std::endl;
        run_gemm_example<float>();
        return 0;
}
