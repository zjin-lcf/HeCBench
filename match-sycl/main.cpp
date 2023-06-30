//**********************************************************//
//   Matching test code by Marten Bjorkman aka Celebrandil  //
//                                                          //
//   The code includes an example of gradual optimization   //
//   of a kernel for matching two sets of 16K 128D points.  //
//**********************************************************//

#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>  // std::align
#include <chrono>
#include <sycl/sycl.hpp>

using float4 = sycl::float4;

#define NPTS (2048*8)
#define NDIM 128

#define M1W  128
#define M2W   16
#define M2H   16
#define M5W   16
#define M5H   16
#define M5R    4
#define M7W   32
#define M7H   32
#define M7R    4

// serial execution on a host
void MatchC1(float *h_pts1, float *h_pts2, float *h_score, int *h_index)
{
  std::memset(h_score, 0, sizeof(float)*NPTS);
  for (int p1=0;p1<NPTS;p1++) {
    for (int p2=0;p2<NPTS;p2++) {
      float score = 0.0f;
      for (int d=0;d<NDIM;d++)
	score += h_pts1[p1*NDIM + d]*h_pts2[p2*NDIM + d];
      if (score>h_score[p1]) {
	h_score[p1] = score;
	h_index[p1] = p2;
      }
    }
  }
}

// verify host and device results
void CheckMatches(int *h_index, int *h_index2, float *h_score, float *h_score2)
{
  int ndiff = 0;
  for (int i=0;i<NPTS;i++) {
    ndiff += (h_index[i] != h_index2[i]);
#ifdef DEBUG
    if (h_index[i] != h_index2[i])
      std::cout << "  " << i << " " << h_index[i] << " " << h_index2[i] << " "
                << h_score[i] << " " << h_score2[i] << std::endl;
#endif
  }
  std::cout << "Number of incorrect matches: " << ndiff << std::endl;
}


void Match1(sycl::nd_item<1> &item,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int p1 = item.get_global_id(0);
  float max_score = 0.0f;
  int index = -1;

  for (int p2=0;p2<NPTS;p2++) {
    float score = 0.0f;
    for (int d=0;d<NDIM;d++)
      score += d_pts1[p1*NDIM + d]*d_pts2[p2*NDIM + d];
    if (score>max_score) {
      max_score = score;
      index = p2;
    }
  }

  d_score[p1] = max_score;
  d_index[p1] = index;
}

void Match2(sycl::nd_item<2> &item,
                  float *__restrict buffer1,
                  float *__restrict buffer2,
                  float *__restrict scores,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int idx = tx + M2W*ty;
  int bp1 = M2W*item.get_group(1);
  if (ty<M2W)
    for (int d=tx;d<NDIM;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*NDIM + d] = d_pts1[(bp1 + j)*NDIM + d];
  item.barrier(sycl::access::fence_space::local_space);

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM;d+=M2W)
      buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d];
    item.barrier(sycl::access::fence_space::local_space);

    float score = 0.0f;
    for (int d=0;d<NDIM;d++)
      score += buffer1[tx*NDIM + d]*buffer2[ty*NDIM + d];
    scores[idx] = score;
    item.barrier(sycl::access::fence_space::local_space);

    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


void Match3(sycl::nd_item<2> &item,
                  float *__restrict buffer1,
                  float *__restrict buffer2,
                  float *__restrict scores,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int idx = tx + M2W*ty;
  int bp1 = M2W*item.get_group(1);
  if (ty<M2W)
    for (int d=tx;d<NDIM;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*(NDIM + 1) + d] = d_pts1[(bp1 + j)*NDIM + d];
  item.barrier(sycl::access::fence_space::local_space);

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM;d+=M2W)
      buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d];
    item.barrier(sycl::access::fence_space::local_space);

    float score = 0.0f;
    for (int d=0;d<NDIM;d++)
      score += buffer1[tx*(NDIM + 1) + d]*buffer2[ty*NDIM + d];
    scores[idx] = score;
    item.barrier(sycl::access::fence_space::local_space);

    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


void Match4(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
                  float *__restrict scores,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int idx = tx + M2W*ty;
  int bp1 = M2W*item.get_group(1);
  if (ty<M2W)
    for (int d=tx;d<NDIM/4;d+=M2W)
      for (int j=ty;j<M2W;j+=M2H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
  item.barrier(sycl::access::fence_space::local_space);

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M2H) {
    for (int d=tx;d<NDIM/4;d+=M2W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    float score = 0.0f;
    for (int d=0;d<NDIM/4;d++) {
      float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
      float4 v2 = buffer2[ty*(NDIM/4) + d];
      score += v1.x()*v2.x(); score += v1.y()*v2.y();
      score += v1.z()*v2.z(); score += v1.w()*v2.w();
    }
    scores[idx] = score;
    item.barrier(sycl::access::fence_space::local_space);

    if (ty==0) {
      for (int i=0;i<M2H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M2W + tx];
	  index = bp2 + i;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

void Match5(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
                  float *__restrict scores,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M5W*item.get_group(1);
  if (ty<M5W)
    for (int d=tx;d<NDIM/4;d+=M5W)
      for (int j=ty;j<M5W;j+=M5H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
  item.barrier(sycl::access::fence_space::local_space);

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M5H) {
    for (int d=tx;d<NDIM/4;d+=M5W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    if (ty<M5H/M5R) {
      float score[M5R];
      for (int dy=0;dy<M5R;dy++)
	score[dy] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
	for (int dy=0;dy<M5R;dy++) {
	  float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];
	  score[dy] += v1.x()*v2.x(); score[dy] += v1.y()*v2.y();
	  score[dy] += v1.z()*v2.z(); score[dy] += v1.w()*v2.w();
	}
      }
      for (int dy=0;dy<M5R;dy++)
	scores[tx + M5W*(M5R*ty + dy)] = score[dy];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (ty==0) {
      for (int i=0;i<M5H;i++) {
	if (scores[i*M2W + tx]>max_score) {
	  max_score = scores[i*M5W + tx];
	  index = bp2 + i;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if (ty==0) {
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}


void Match6(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M5W*item.get_group(1);
  if (ty<M5W)
    for (int d=tx;d<NDIM/4;d+=M5W)
      for (int j=ty;j<M5W;j+=M5H)
	buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M5H) {
    for (int d=tx;d<NDIM/4;d+=M5W)
      buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    if (ty<M5H/M5R) {
      float score[M5R];
      for (int dy=0;dy<M5R;dy++)
	score[dy] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
	for (int dy=0;dy<M5R;dy++) {
	  float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];
	  score[dy] += v1.x()*v2.x(); score[dy] += v1.y()*v2.y();
	  score[dy] += v1.z()*v2.z(); score[dy] += v1.w()*v2.w();
	}
      }
      for (int dy=0;dy<M5R;dy++) {
	if (score[dy]>max_score) {
	  max_score = score[dy];
	  index = bp2 + M5R*ty + dy;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M5W*M5H/M5R];
  if (ty<M5H/M5R) {
    scores[ty*M5W + tx] = max_score;
    indices[ty*M5W + tx] = index;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (ty==0) {
    max_score = scores[tx];
    index = indices[tx];
    for (int y=0;y<M5H/M5R;y++)
      if (scores[y*M5W + tx]>max_score) {
	max_score = scores[y*M5W + tx];
	index = indices[y*M5W + tx];
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

void Match7(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M7W*item.get_group(1);
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score = 0.0f;
  int index = -1;
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R)
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    float score[M7R];
    for (int dy=0;dy<M7R;dy++)
      score[dy] = 0.0f;
    for (int d=0;d<NDIM/4;d++) {
      float4 v1 = buffer1[tx*NDIM/4 + (d + tx)%(NDIM/4)];
      for (int dy=0;dy<M7R;dy++) {
	float4 v2 = buffer2[(M7R*ty + dy)*(NDIM/4) + d];
	score[dy] += v1.x()*v2.x();
        score[dy] += v1.y()*v2.y();
	score[dy] += v1.z()*v2.z();
        score[dy] += v1.w()*v2.w();
      }
    }
    for (int dy=0;dy<M7R;dy++) {
      if (score[dy]>max_score) {
	max_score = score[dy];
	index = bp2 + M7R*ty + dy;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  scores[ty*M7W + tx] = max_score;
  indices[ty*M7W + tx] = index;
  item.barrier(sycl::access::fence_space::local_space);

  if (ty==0) {
    max_score = scores[tx];
    index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx];
	index = indices[y*M7W + tx];
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

void Match8(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M7W*item.get_group(1);
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

#define NRX 2
  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R)
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    if (idx<M7W*M7H/M7R/NRX) {
      float score[M7R][NRX];
      for (int dy=0;dy<M7R;dy++)
	for (int i=0;i<NRX;i++)
	  score[dy][i] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1[NRX];
	for (int i=0;i<NRX;i++)
	  v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
	for (int dy=0;dy<M7R;dy++) {
	  float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];
	  for (int i=0;i<NRX;i++) {
	    score[dy][i] += v1[i].x()*v2.x();
	    score[dy][i] += v1[i].y()*v2.y();
	    score[dy][i] += v1[i].z()*v2.z();
	    score[dy][i] += v1[i].w()*v2.w();
	  }
	}
      }
      for (int dy=0;dy<M7R;dy++) {
	for (int i=0;i<NRX;i++) {
	  if (score[dy][i]>max_score[i]) {
	    max_score[i] = score[dy][i];
	    index[i] = bp2 + M7R*iy + dy;
	  }
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx];
	index = indices[y*M7W + tx];
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

void Match9(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M7W*item.get_group(1);
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R/NRX)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    for (int d=tx;d<NDIM/4;d+=M7W)
      for (int j=ty;j<M7H;j+=M7H/M7R/NRX)
	buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);

    float score[M7R][NRX];
    for (int dy=0;dy<M7R;dy++)
      for (int i=0;i<NRX;i++)
	score[dy][i] = 0.0f;
    for (int d=0;d<NDIM/4;d++) {
      float4 v1[NRX];
      for (int i=0;i<NRX;i++)
	v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
      for (int dy=0;dy<M7R;dy++) {
	float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];
	for (int i=0;i<NRX;i++) {
	  score[dy][i] += v1[i].x()*v2.x();
	  score[dy][i] += v1[i].y()*v2.y();
	  score[dy][i] += v1[i].z()*v2.z();
	  score[dy][i] += v1[i].w()*v2.w();
	}
      }
    }
    for (int dy=0;dy<M7R;dy++) {
      for (int i=0;i<NRX;i++) {
	if (score[dy][i]>max_score[i]) {
	  max_score[i] = score[dy][i];
	  index[i] = bp2 + M7R*iy + dy;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx];
	index = indices[y*M7W + tx];
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

void Match10(sycl::nd_item<2> &item,
                 float4 *__restrict buffer1,
                 float4 *__restrict buffer2,
            const float *__restrict d_pts1,
            const float *__restrict d_pts2,
                  float *__restrict d_score,
                    int *__restrict d_index)
{
#define NRX 2
#define NUM (NRX*M7R)                       // 32*8 threads
  int tx = item.get_local_id(1);
  int ty = item.get_local_id(0);
  int bp1 = M7W*item.get_group(1);
  for (int d=tx;d<NDIM/4;d+=M7W)
    for (int j=ty;j<M7W;j+=M7H/M7R)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];

  float max_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<NPTS;bp2+=M7H) {
    float score[M7R][NRX];
    for (int dy=0;dy<M7R;dy++)
      for (int i=0;i<NRX;i++)
	score[dy][i] = 0.0f;

    int d = (idx%NUM);
    int j = (idx/NUM);
    buffer2[j*NUM + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
    item.barrier(sycl::access::fence_space::local_space);
    for (int dp=0;dp<NDIM/4;dp+=NUM) {
      float4 temp;
      if (dp<(NDIM/4-NUM))
	temp = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + dp + d + NUM];

      if (idx<M7W*M7H/M7R/NRX) {
	for (int d=0;d<NUM;d++) {
	  float4 v1[NRX];
#pragma unroll
	  for (int i=0;i<NRX;i++)
	    v1[i] = buffer1[(((M7W/NRX)*i + ix)<<5) + ((dp + d + (M7W/NRX)*i + ix)&31)];
	  //v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (dp + d + (M7W/NRX)*i + ix)%(NDIM/4)];
#pragma unroll
	  for (int dy=0;dy<M7R;dy++) {
	    float4 v2 = buffer2[(M7R*iy + dy)*NUM + d];
#pragma unroll
	    for (int i=0;i<NRX;i++) {
	      score[dy][i] += v1[i].x()*v2.x();
	      score[dy][i] += v1[i].y()*v2.y();
	      score[dy][i] += v1[i].z()*v2.z();
	      score[dy][i] += v1[i].w()*v2.w();
	    }
	  }
	}
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (dp<(NDIM/4-NUM)) {
	buffer2[j*NUM + d] = temp;
	item.barrier(sycl::access::fence_space::local_space);
      }
    }
    for (int dy=0;dy<M7R;dy++) {
      for (int i=0;i<NRX;i++) {
	if (score[dy][i]>max_score[i]) {
	  max_score[i] = score[dy][i];
	  index[i] = bp2 + M7R*iy + dy;
	}
      }
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  float *scores = (float*)buffer1;
  int *indices = (int*)&scores[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (ty==0) {
    float max_score = scores[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (scores[y*M7W + tx]>max_score) {
	max_score = scores[y*M7W + tx];
	index = indices[y*M7W + tx];
      }
    d_score[bp1 + tx] = max_score;
    d_index[bp1 + tx] = index;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t space = sizeof(float)*NPTS*NDIM*2 + 8;
  std::vector<float> data(NPTS*NDIM*2 + 8);
  void *ptr = (void*)&data[0];
  float *h_pts1 = (float*)std::align(32, sizeof(float)*NPTS*NDIM, ptr, space);
  ptr = (void*)&data[NPTS*NDIM];
  float *h_pts2 = (float*)std::align(32, sizeof(float)*NPTS*NDIM, ptr, space);
  std::vector<int> h_index(NPTS);
  std::vector<float> h_score(NPTS);
  std::vector<int> h_index2(NPTS);
  std::vector<float> h_score2(NPTS);

  std::cout << std::endl;
  int psize = sizeof(float)*NPTS;
  std::cout << "Data size:   " << 2.0*psize*NDIM/1024/1024 << " MB" << std::endl;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_pts1  = sycl::malloc_device<float>(NPTS*NDIM, q);
  float *d_pts2  = sycl::malloc_device<float>(NPTS*NDIM, q);
    int *d_index = sycl::malloc_device<  int>(NPTS, q);
  float *d_score = sycl::malloc_device<float>(NPTS, q);

  for (int i=0;i<NPTS;i++) {
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int d=0;d<NDIM;d++) {
      sum1 += h_pts1[i*NDIM + d] = (float)rand()/RAND_MAX;
      sum2 += h_pts2[i*NDIM + d] = (float)rand()/RAND_MAX;
    }
    sum1 = sqrt(NDIM)/sum1;
    sum2 = sqrt(NDIM)/sum2;
    for (int d=0;d<NDIM;d++) {
      h_pts1[i*NDIM + d] *= sum1;
      h_pts2[i*NDIM + d] *= sum2;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  MatchC1(h_pts1, h_pts2, h_score.data(), h_index.data());
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  auto delay = elapsed_seconds.count() * 1000;
  std::cout << "MatchCPU1:   " << delay << " ms  "
            << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;

  q.memcpy(d_pts1, h_pts1, psize*NDIM);
  q.memcpy(d_pts2, h_pts2, psize*NDIM);
  q.wait();

  sycl::range<1> gws1 (NPTS);
  sycl::range<1> lws1 (M1W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class k1>(
        sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
        Match1(item, d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU1:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws2 (M2H, NPTS/M2W);
  sycl::range<2> lws2 (M2H, M2W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> buffer1(sycl::range<1>(M2W*NDIM), cgh);
      sycl::local_accessor<float, 1> buffer2(sycl::range<1>(M2H*NDIM), cgh);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(M2H*M2W), cgh);
      cgh.parallel_for<class k2>(
        sycl::nd_range<2>(gws2, lws2), [=] (sycl::nd_item<2> item) {
        Match2(item, buffer1.get_pointer(), buffer2.get_pointer(), scores.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU2:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws3 (M2H, NPTS/M2W);
  sycl::range<2> lws3 (M2H, M2W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> buffer1(sycl::range<1>(M2W*(NDIM+1)), cgh);
      sycl::local_accessor<float, 1> buffer2(sycl::range<1>(M2H*NDIM), cgh);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(M2H*M2W), cgh);
      cgh.parallel_for<class k3>(
        sycl::nd_range<2>(gws3, lws3), [=] (sycl::nd_item<2> item) {
        Match3(item, buffer1.get_pointer(), buffer2.get_pointer(), scores.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU3:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws4 (M2H, NPTS);
  sycl::range<2> lws4 (M2H, M2W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M2W*(NDIM/4+1)), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M2H*NDIM/4), cgh);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(M2H*M2W), cgh);
      cgh.parallel_for<class k4>(
        sycl::nd_range<2>(gws4, lws4), [=] (sycl::nd_item<2> item) {
        Match4(item, buffer1.get_pointer(), buffer2.get_pointer(), scores.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU4:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws5 (M5H, NPTS);
  sycl::range<2> lws5 (M5H, M5W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M5W*(NDIM/4+1)), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M5H*NDIM/4), cgh);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(M5H*M5W), cgh);
      cgh.parallel_for<class k5>(
        sycl::nd_range<2>(gws5, lws5), [=] (sycl::nd_item<2> item) {
        Match5(item, buffer1.get_pointer(), buffer2.get_pointer(), scores.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU5:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws6 (M5H, NPTS);
  sycl::range<2> lws6 (M5H, M5W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M5W*(NDIM/4+1)), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M5H*NDIM/4), cgh);
      cgh.parallel_for<class k6>(
        sycl::nd_range<2>(gws6, lws6), [=] (sycl::nd_item<2> item) {
        Match6(item, buffer1.get_pointer(), buffer2.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU6:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws7 (M7H/M7R, NPTS);
  sycl::range<2> lws7 (M7H/M7R, M7W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M7W*NDIM/4), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M7H*NDIM/4), cgh);
      cgh.parallel_for<class k7>(
        sycl::nd_range<2>(gws7, lws7), [=] (sycl::nd_item<2> item) {
        Match7(item, buffer1.get_pointer(), buffer2.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU7:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws8 (M7H/M7R, NPTS);
  sycl::range<2> lws8 (M7H/M7R, M7W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M7W*NDIM/4), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M7H*NDIM/4), cgh);
      cgh.parallel_for<class k8>(
        sycl::nd_range<2>(gws8, lws8), [=] (sycl::nd_item<2> item) {
        Match8(item, buffer1.get_pointer(), buffer2.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU8:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws9 (M7H/M7R/2, NPTS);
  sycl::range<2> lws9 (M7H/M7R/2, M7W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M7W*NDIM/4), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M7H*NDIM/4), cgh);
      cgh.parallel_for<class k9>(
        sycl::nd_range<2>(gws9, lws9), [=] (sycl::nd_item<2> item) {
        Match9(item, buffer1.get_pointer(), buffer2.get_pointer(),
               d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU9:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::range<2> gws10 (M7H/M7R, NPTS);
  sycl::range<2> lws10 (M7H/M7R, M7W);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++)
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float4, 1> buffer1(sycl::range<1>(M7W*NDIM/4), cgh);
      sycl::local_accessor<float4, 1> buffer2(sycl::range<1>(M7H*NUM), cgh);
      cgh.parallel_for<class k10>(sycl::nd_range<2>(gws10, lws10), [=] (sycl::nd_item<2> item) {
        Match10(item, buffer1.get_pointer(), buffer2.get_pointer(),
                d_pts1, d_pts2, d_score, d_index);
      });
    });
  q.wait();
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU10:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  q.memcpy(h_index2.data(), d_index, psize);
  q.memcpy(h_score2.data(), d_score, psize);
  q.wait();
  CheckMatches(h_index.data(), h_index2.data(), h_score.data(), h_score2.data());

  sycl::free(d_pts1, q);
  sycl::free(d_pts2, q);
  sycl::free(d_index, q);
  sycl::free(d_score, q);
  return 0;
}
