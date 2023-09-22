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

// definition to use efficient __mul24 intrinsic

#define INDEX(i,j,j_off)  (i + sycl::mul24(j,j_off))

#define syncthreads() item.barrier(sycl::access::fence_space::local_space)

// device code

void laplace3d(
  sycl::nd_item<2> &item,
  int NX, int NY, int NZ, int pitch, 
  const float *__restrict d_u1,
        float *__restrict d_u2,
        float *__restrict u1)
{
  int   indg, indg_h, indg0;
  int   i, j, k, ind, ind_h, halo, active;
  float u2, sixth=1.0f/6.0f;

  int NXM1 = NX-1;
  int NYM1 = NY-1;
  int NZM1 = NZ-1;

  //
  // first set up indices for halos
  //
  const int threadIdx_x = item.get_local_id(1);
  const int threadIdx_y = item.get_local_id(0);
  const int blockIdx_x = item.get_group(1);
  const int blockIdx_y = item.get_group(0);

  k    = threadIdx_x + threadIdx_y*BLOCK_X;
  halo = k < 2*(BLOCK_X+BLOCK_Y+2);

  if (halo) {
    if (threadIdx_y<2) {               // y-halos (coalesced)
      i = threadIdx_x;
      j = threadIdx_y*(BLOCK_Y+1) - 1;
    }
    else {                             // x-halos (not coalesced)
      i = (k%2)*(BLOCK_X+1) - 1;
      j =  k/2 - BLOCK_X - 1;
    }

    ind_h  = INDEX(i+1,j+1,JOFF) + KOFF;

    i      = INDEX(i,blockIdx_x,BLOCK_X);   // global indices
    j      = INDEX(j,blockIdx_y,BLOCK_Y);
    indg_h = INDEX(i,j,pitch);

    halo   =  (i>=0) && (i<NX) && (j>=0) && (j<NY);
  }

  //
  // then set up indices for main block
  //

  i    = threadIdx_x;
  j    = threadIdx_y;
  ind  = INDEX(i+1,j+1,JOFF) + KOFF;

  i    = INDEX(i,blockIdx_x,BLOCK_X);     // global indices
  j    = INDEX(j,blockIdx_y,BLOCK_Y);
  indg = INDEX(i,j,pitch);

  active = (i<NX) && (j<NY);

  //
  // read initial plane of u1 array
  //

  if (active) u1[ind+KOFF] = d_u1[indg];
  if (halo) u1[ind_h+KOFF] = d_u1[indg_h];

  //
  // loop over k-planes
  //

  for (k=0; k<NZ; k++) {

    // move two planes down and read in new plane k+1

    if (active) {
      indg0 = indg;
      indg  = INDEX(indg,NY,pitch);
      u1[ind-KOFF] = u1[ind];
      u1[ind]      = u1[ind+KOFF];
      if (k<NZM1)
        u1[ind+KOFF] = d_u1[indg];
    }

    if (halo) {
      indg_h = INDEX(indg_h,NY,pitch);
      u1[ind_h-KOFF] = u1[ind_h];
      u1[ind_h]      = u1[ind_h+KOFF];
      if (k<NZM1)
        u1[ind_h+KOFF] = d_u1[indg_h];
    }

    syncthreads();

  //
  // perform Jacobi iteration to set values in u2
  //

    if (active) {
      if (i==0 || i==NXM1 || j==0 || j==NYM1 || k==0 || k==NZM1) {
        u2 = u1[ind];          // Dirichlet b.c.'s
      }
      else {
        u2 = ( u1[ind-IOFF] + u1[ind+IOFF]
             + u1[ind-JOFF] + u1[ind+JOFF]
             + u1[ind-KOFF] + u1[ind+KOFF] ) * sixth;
      }
      d_u2[indg0] = u2;
    }

    syncthreads();

  }
}
