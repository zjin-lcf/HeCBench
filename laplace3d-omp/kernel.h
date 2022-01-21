//
// Notes:
//
// 1) strategy: one thread per node in the 2D block;
//    after initialisation it marches in the k-direction
//    working with 3 planes of data at a time
//
// 2) each thread also loads in data for at most one halo node;
//    assumes the number of halo nodes is not more than the
//    number of interior nodes
//
// 3) corner halo nodes are included because they are needed
//    for more general applications with cross-derivatives
//
// 4) could try double-buffering in the future fairly easily
//

// define kernel block size

#define BLOCK_X 32
#define BLOCK_Y 8

#define IOFF  1
#define JOFF (BLOCK_X+2)
#define KOFF (BLOCK_X+2)*(BLOCK_Y+2)

#define INDEX(i,j,j_off)  (i + (j) * (j_off))

// device code

void laplace3d(
  int NX, int NY, int NZ, int pitch, 
  const float *__restrict u1,
        float *__restrict u2)
{
  #pragma omp target teams distribute parallel for collapse(3) thread_limit(BLOCK_X*BLOCK_Y)
  for (int k=0; k<NZ; k++) {
    for (int j=0; j<NY; j++) {
      for (int i=0; i<NX; i++) {   // i loop innermost for sequential memory access
	int ind = i + j*NX + k*NX*NY;

        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1) {
          u2[ind] = u1[ind];          // Dirichlet b.c.'s
        }
        else {
          u2[ind] = ( u1[ind-1    ] + u1[ind+1    ]
                    + u1[ind-NX   ] + u1[ind+NX   ]
                    + u1[ind-NX*NY] + u1[ind+NX*NY] ) * 1.0f/6.0f;
        }
      }
    }
  }
}
