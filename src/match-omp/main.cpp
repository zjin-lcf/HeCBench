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
#include <omp.h>

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

#define NRX 2
#define NUM (NRX*M7R)                       // 32*8 threads

typedef struct __attribute__((__aligned__(16)))
{
  float x, y, z, w;
} float4;

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
    if (h_index[i] != h_index2[i])
      std::cout << "  " << i << " " << h_index[i] << " " << h_index2[i] << " " << h_score[i] << " " << h_score2[i] << std::endl;
  }
  std::cout << "Number of incorrect matches: " << ndiff << std::endl;
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

  float *d_pts1 = h_pts1;
  float *d_pts2 = h_pts2;
    int *d_index = h_index2.data();
  float *d_score = h_score2.data();
#pragma omp target data map (to: d_pts1[0:NPTS*NDIM], d_pts2[0:NPTS*NDIM]) \
                        map (alloc: d_index[0:NPTS], d_score[0:NPTS])
{
  auto start = std::chrono::high_resolution_clock::now();
  MatchC1(h_pts1, h_pts2, h_score.data(), h_index.data());
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_seconds = end - start;
  auto delay = elapsed_seconds.count() * 1000;
  std::cout << "MatchCPU1:   " << delay << " ms  "
            << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit (M1W)
    for (int p1 = 0; p1 < NPTS; p1++) { 
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
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU1:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M2W) thread_limit(M2W*M2H)
    {
      float buffer1[M2W*NDIM];  
      float buffer2[M2H*NDIM];  
      float scores[M2W*M2H];    
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M2W;
        int ty = omp_get_thread_num() / M2W;
        int idx = tx + M2W*ty;
        int bp1 = M2W*omp_get_team_num();
        if (ty<M2W)
          for (int d=tx;d<NDIM;d+=M2W)
            for (int j=ty;j<M2W;j+=M2H)
              buffer1[j*NDIM + d] = d_pts1[(bp1 + j)*NDIM + d];   
        #pragma omp barrier
        
        float max_score = 0.0f;
        int index = -1;
        for (int bp2=0;bp2<NPTS;bp2+=M2H) {
          for (int d=tx;d<NDIM;d+=M2W)
            buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d]; 
          #pragma omp barrier

          float score = 0.0f;
          for (int d=0;d<NDIM;d++) 
            score += buffer1[tx*NDIM + d]*buffer2[ty*NDIM + d];   
          scores[idx] = score;
          #pragma omp barrier
          
          if (ty==0) {
            for (int i=0;i<M2H;i++) {
              if (scores[i*M2W + tx]>max_score) {
                max_score = scores[i*M2W + tx];
                index = bp2 + i;
              }
            }
          }
          #pragma omp barrier
        }
        
        if (ty==0) {
          d_score[bp1 + tx] = max_score;
          d_index[bp1 + tx] = index;
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU2:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M2W) thread_limit(M2W*M2H)
    {
      float buffer1[M2W*(NDIM + 1)]; 
      float buffer2[M2H*NDIM];
      float scores[M2W*M2H];
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M2W;
        int ty = omp_get_thread_num() / M2W;
        int idx = tx + M2W*ty;
        int bp1 = M2W*omp_get_team_num();
        if (ty<M2W)
          for (int d=tx;d<NDIM;d+=M2W)
            for (int j=ty;j<M2W;j+=M2H)
              buffer1[j*(NDIM + 1) + d] = d_pts1[(bp1 + j)*NDIM + d]; 
        #pragma omp barrier
        
        float max_score = 0.0f;
        int index = -1;
        for (int bp2=0;bp2<NPTS;bp2+=M2H) {
          for (int d=tx;d<NDIM;d+=M2W)
            buffer2[ty*NDIM + d] = d_pts2[(bp2 + ty)*NDIM + d];
          #pragma omp barrier

          float score = 0.0f;
          for (int d=0;d<NDIM;d++) 
            score += buffer1[tx*(NDIM + 1) + d]*buffer2[ty*NDIM + d]; 
          scores[idx] = score;
          #pragma omp barrier
          
          if (ty==0) {
            for (int i=0;i<M2H;i++) {
              if (scores[i*M2W + tx]>max_score) {
                max_score = scores[i*M2W + tx];
                index = bp2 + i;
              }
            }
          }
          #pragma omp barrier
        }
  
        if (ty==0) {
          d_score[bp1 + tx] = max_score;
          d_index[bp1 + tx] = index;
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU3:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);
  
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M2W) thread_limit(M2W*M2H)
    {
      float4 buffer1[M2W*(NDIM/4 + 1)];  
      float4 buffer2[M2H*NDIM/4];        
      float scores[M2W*M2H];
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M2W;
        int ty = omp_get_thread_num() / M2W;
        int idx = tx + M2W*ty;
        int bp1 = M2W*omp_get_team_num();
        if (ty<M2W)
          for (int d=tx;d<NDIM/4;d+=M2W)
            for (int j=ty;j<M2W;j+=M2H)
              buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d]; 
        #pragma omp barrier
        
        float max_score = 0.0f;
        int index = -1;
        for (int bp2=0;bp2<NPTS;bp2+=M2H) {
          for (int d=tx;d<NDIM/4;d+=M2W)
            buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d]; 
          #pragma omp barrier

          float score = 0.0f;
          for (int d=0;d<NDIM/4;d++) {
            float4 v1 = buffer1[tx*(NDIM/4 + 1) + d]; 
            float4 v2 = buffer2[ty*(NDIM/4) + d];     
            score += v1.x*v2.x; score += v1.y*v2.y;
            score += v1.z*v2.z; score += v1.w*v2.w;
          }
          scores[idx] = score;
          #pragma omp barrier
          
          if (ty==0) {
            for (int i=0;i<M2H;i++) {
              if (scores[i*M2W + tx]>max_score) {
                max_score = scores[i*M2W + tx];
                index = bp2 + i;
              }
            }
          }
          #pragma omp barrier
        }
        
        if (ty==0) {
          d_score[bp1 + tx] = max_score;
          d_index[bp1 + tx] = index;
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU4:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);
  
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M5W) thread_limit(M5W*M5H)
    {
      float4 buffer1[M5W*(NDIM/4 + 1)]; 
      float4 buffer2[M5H*NDIM/4];       
      float scores[M5W*M5H];
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M5W;
        int ty = omp_get_thread_num() / M5W;
        int bp1 = M5W*omp_get_team_num();
        if (ty<M5W)
          for (int d=tx;d<NDIM/4;d+=M5W)
            for (int j=ty;j<M5W;j+=M5H)
              buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
        #pragma omp barrier
        
        float max_score = 0.0f;
        int index = -1;
        for (int bp2=0;bp2<NPTS;bp2+=M5H) {
          for (int d=tx;d<NDIM/4;d+=M5W)
            buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
          #pragma omp barrier

          if (ty<M5H/M5R) {  
            float score[M5R];                                    
            for (int dy=0;dy<M5R;dy++)
              score[dy] = 0.0f;
            for (int d=0;d<NDIM/4;d++) {
              float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
              for (int dy=0;dy<M5R;dy++) {
                float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];    
                score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
                score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
              }
            }
            for (int dy=0;dy<M5R;dy++)
              scores[tx + M5W*(M5R*ty + dy)] = score[dy];
          }
          #pragma omp barrier
          
          if (ty==0) {
            for (int i=0;i<M5H;i++) {
              if (scores[i*M2W + tx]>max_score) {
                max_score = scores[i*M5W + tx];
                index = bp2 + i;
              }
            }
          }
          #pragma omp barrier
        }

        if (ty==0) {
          d_score[bp1 + tx] = max_score;
          d_index[bp1 + tx] = index;
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU5:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);
  
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M5W) thread_limit(M5W*M5H)
    {
      float4 buffer1[M5W*(NDIM/4 + 1)]; 
      float4 buffer2[M5H*NDIM/4];       
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M5W;
        int ty = omp_get_thread_num() / M5W;
        int bp1 = M5W*omp_get_team_num();
        if (ty<M5W)
          for (int d=tx;d<NDIM/4;d+=M5W)
            for (int j=ty;j<M5W;j+=M5H)
              buffer1[j*(NDIM/4 + 1) + d] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
        
        float max_score = 0.0f;
        int index = -1;    
        for (int bp2=0;bp2<NPTS;bp2+=M5H) {
          for (int d=tx;d<NDIM/4;d+=M5W)
            buffer2[ty*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + ty)*(NDIM/4) + d];
          #pragma omp barrier

          if (ty<M5H/M5R) {  
            float score[M5R];                                    
            for (int dy=0;dy<M5R;dy++)
              score[dy] = 0.0f;
            for (int d=0;d<NDIM/4;d++) {
              float4 v1 = buffer1[tx*(NDIM/4 + 1) + d];
              for (int dy=0;dy<M5R;dy++) {
                float4 v2 = buffer2[(M5R*ty + dy)*(NDIM/4) + d];    
                score[dy] += v1.x*v2.x; score[dy] += v1.y*v2.y;
                score[dy] += v1.z*v2.z; score[dy] += v1.w*v2.w;
              }
            }
            for (int dy=0;dy<M5R;dy++) {
              if (score[dy]>max_score) {   
                max_score = score[dy];     
                index = bp2 + M5R*ty + dy;               
              }
            }
          }
          #pragma omp barrier
        }

        float *scores = (float*)buffer1;
        int *indices = (int*)&scores[M5W*M5H/M5R];
        if (ty<M5H/M5R) {
          scores[ty*M5W + tx] = max_score;  
          indices[ty*M5W + tx] = index;     
        }
        #pragma omp barrier
        
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
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU6:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M7W) thread_limit(M7W*M7H/M7R)
    {
      float4 buffer1[M7W*NDIM/4]; 
      float4 buffer2[M7H*NDIM/4];       
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M7W;
        int ty = omp_get_thread_num() / M7W;
        int bp1 = M7W*omp_get_team_num();
        for (int d=tx;d<NDIM/4;d+=M7W)
          for (int j=ty;j<M7W;j+=M7H/M7R)      
            buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)d_pts1)[(bp1 + j)*(NDIM/4) + d];
        
        float max_score = 0.0f;
        int index = -1;    
        for (int bp2=0;bp2<NPTS;bp2+=M7H) {
          for (int d=tx;d<NDIM/4;d+=M7W)
            for (int j=ty;j<M7H;j+=M7H/M7R)       
              buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
          #pragma omp barrier

          float score[M7R];                                    
          for (int dy=0;dy<M7R;dy++)
            score[dy] = 0.0f;
          for (int d=0;d<NDIM/4;d++) {
            float4 v1 = buffer1[tx*NDIM/4 + (d + tx)%(NDIM/4)];
            for (int dy=0;dy<M7R;dy++) {
              float4 v2 = buffer2[(M7R*ty + dy)*(NDIM/4) + d];    
              score[dy] += v1.x*v2.x;
              score[dy] += v1.y*v2.y;
              score[dy] += v1.z*v2.z;
              score[dy] += v1.w*v2.w;
            }
          }
          for (int dy=0;dy<M7R;dy++) {
            if (score[dy]>max_score) {   
              max_score = score[dy];     
              index = bp2 + M7R*ty + dy;               
            }
          }
          #pragma omp barrier
        }

        float *scores = (float*)buffer1;
        int *indices = (int*)&scores[M7W*M7H/M7R];
        scores[ty*M7W + tx] = max_score;  
        indices[ty*M7W + tx] = index;     
        #pragma omp barrier
        
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
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU7:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M7W) thread_limit(M7W*M7H/M7R)
    {
      float4 buffer1[M7W*NDIM/4]; 
      float4 buffer2[M7H*NDIM/4];       
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M7W;
        int ty = omp_get_thread_num() / M7W;
        int bp1 = M7W*omp_get_team_num();
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
          for (int d=tx;d<NDIM/4;d+=M7W)
            for (int j=ty;j<M7H;j+=M7H/M7R)       
              buffer2[j*NDIM/4 + d] = ((float4*)d_pts2)[(bp2 + j)*(NDIM/4) + d];
          #pragma omp barrier

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
                  score[dy][i] += v1[i].x*v2.x;
                  score[dy][i] += v1[i].y*v2.y;
                  score[dy][i] += v1[i].z*v2.z;
                  score[dy][i] += v1[i].w*v2.w;
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
          #pragma omp barrier
        }

        float *scores = (float*)buffer1;
        int *indices = (int*)&scores[M7W*M7H/M7R];
        if (idx<M7W*M7H/M7R/NRX) {
          for (int i=0;i<NRX;i++) {
            scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
            indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
          }
        }
        #pragma omp barrier
        
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
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU8:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M7W) thread_limit(M7W*M7H/M7R/2)
    {
      float4 buffer1[M7W*NDIM/4]; 
      float4 buffer2[M7H*NDIM/4];       
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M7W;
        int ty = omp_get_thread_num() / M7W;
        int bp1 = M7W*omp_get_team_num();
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
          #pragma omp barrier

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
                score[dy][i] += v1[i].x*v2.x;
                score[dy][i] += v1[i].y*v2.y;
                score[dy][i] += v1[i].z*v2.z;
                score[dy][i] += v1[i].w*v2.w;
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
          #pragma omp barrier
        }

        float *scores = (float*)buffer1;
        int *indices = (int*)&scores[M7W*M7H/M7R];
        if (idx<M7W*M7H/M7R/NRX) {
          for (int i=0;i<NRX;i++) {
            scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
            indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
          }
        }
        #pragma omp barrier
        
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
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "Match9:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);


  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams num_teams(NPTS/M7W) thread_limit(M7W*M7H/M7R)
    {
      float4 buffer1[M7W*NDIM/4];    // 32*32
      float4 buffer2[M7H*NUM];       // 32*8
      #pragma omp parallel 
      {
        int tx = omp_get_thread_num() % M7W;
        int ty = omp_get_thread_num() / M7W;
        int bp1 = M7W*omp_get_team_num();
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
          #pragma omp barrier
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
                    score[dy][i] += v1[i].x*v2.x;
                    score[dy][i] += v1[i].y*v2.y;
                    score[dy][i] += v1[i].z*v2.z;
                    score[dy][i] += v1[i].w*v2.w;
                  }
                }
              }
            }
            #pragma omp barrier

            if (dp<(NDIM/4-NUM)) {
              buffer2[j*NUM + d] = temp;
              #pragma omp barrier
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
          #pragma omp barrier
        }

        float *scores = (float*)buffer1;
        int *indices = (int*)&scores[M7W*M7H/M7R];
        if (idx<M7W*M7H/M7R/NRX) {
          for (int i=0;i<NRX;i++) {
            scores[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
            indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
          }
        }
        #pragma omp barrier
        
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
    }
  }
  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  delay = elapsed_seconds.count() * 1000 / repeat;
  std::cout << "MatchGPU10:   " << delay << " ms  " << 2.0*NPTS*NPTS*NDIM/delay/1024/1024 << " Gflops" << std::endl;
  #pragma omp target update from (d_index[0:NPTS])
  #pragma omp target update from (d_score[0:NPTS])
  CheckMatches(h_index.data(), d_index, h_score.data(), d_score);
}

  return 0;
}
