// called by host and device

#pragma omp declare target
inline uchar4 operator^(uchar4 a, uchar4 b)
{
  return {(uchar)(a.x ^ b.x), (uchar)(a.y ^ b.y), 
          (uchar)(a.z ^ b.z), (uchar)(a.w ^ b.w)};
}

inline void operator^=(uchar4 &a, const uchar4 b)
{
  a.x ^= b.x;
  a.y ^= b.y;
  a.z ^= b.z;
  a.w ^= b.w;
}

uchar galoisMultiplication(uchar a, uchar b)
{
    uchar p = 0; 
    for(unsigned int i=0; i < 8; ++i)
    {
        if((b&1) == 1)
        {
            p^=a;
        }
        uchar hiBitSet = (a & 0x80);
        a <<= 1;
        if(hiBitSet == 0x80)
        {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    return p;
}

inline
uchar4 sboxRead(const uchar * SBox, uchar4 block)
{
    return {SBox[block.x], SBox[block.y], SBox[block.z], SBox[block.w]};
}

uchar4 mixColumns(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
    unsigned int bw = 4;

    uchar x, y, z, w;

    x = galoisMultiplication(block[0].x, galiosCoeff[(bw-j)%bw].x);
    y = galoisMultiplication(block[0].y, galiosCoeff[(bw-j)%bw].x);
    z = galoisMultiplication(block[0].z, galiosCoeff[(bw-j)%bw].x);
    w = galoisMultiplication(block[0].w, galiosCoeff[(bw-j)%bw].x);
   
    for(unsigned int k=1; k< 4; ++k)
    {
        x ^= galoisMultiplication(block[k].x, galiosCoeff[(k+bw-j)%bw].x);
        y ^= galoisMultiplication(block[k].y, galiosCoeff[(k+bw-j)%bw].x);
        z ^= galoisMultiplication(block[k].z, galiosCoeff[(k+bw-j)%bw].x);
        w ^= galoisMultiplication(block[k].w, galiosCoeff[(k+bw-j)%bw].x);
    }
    
    return {x, y, z, w};
}

uchar4 shiftRows(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        //r.xyzw() = r.yzwx();
        uchar x = r.x;
        uchar y = r.y;
        uchar z = r.z;
        uchar w = r.w;
        r = {y,z,w,x};
    }
    return r;
}

uchar4 shiftRowsInv(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        // r = r.wxyz();
        uchar x = r.x;
        uchar y = r.y;
        uchar z = r.z;
        uchar w = r.w;
        r = {w,x,y,z};
    }
    return r;
}
#pragma omp end declare target

void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                const uint     width , 
                const uint     height , 
                const uint     rounds )
                                
{
   const unsigned int teams = width*height/16;
   const unsigned int threads = 4;
                                
   #pragma omp target teams num_teams(teams) thread_limit(threads)
   {
    uchar4 block0[4];
    uchar4 block1[4];
    #pragma omp parallel 
    {
    unsigned int bx = omp_get_team_num() % (width/4);
    unsigned int by = omp_get_team_num() / (width/4);
 
    //unsigned int localIdx = threadIdx.x;
    unsigned int localIdy = omp_get_thread_num();
    
    unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
    unsigned int localIndex  = localIdy;

    uchar4 galiosCoeff[4];
    galiosCoeff[0] = {2, 0, 0, 0};
    galiosCoeff[1] = {3, 0, 0, 0};
    galiosCoeff[2] = {1, 0, 0, 0};
    galiosCoeff[3] = {1, 0, 0, 0};

    block0[localIndex]  = input[globalIndex];
    
    block0[localIndex] ^= roundKey[localIndex];

    for(unsigned int r=1; r < rounds; ++r)
    {
        block0[localIndex] = sboxRead(SBox, block0[localIndex]);

        block0[localIndex] = shiftRows(block0[localIndex], localIndex); 
       
        #pragma omp barrier
        block1[localIndex]  = mixColumns(block0, galiosCoeff, localIndex); 
        
        #pragma omp barrier
        block0[localIndex] = block1[localIndex]^roundKey[r*4 + localIndex];
    }  
    block0[localIndex] = sboxRead(SBox, block0[localIndex]);
  
    block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

    output[globalIndex] =  block0[localIndex]^roundKey[(rounds)*4 + localIndex];
    }
  }
}

void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                const  uint    width , 
                const  uint    height , 
                const  uint    rounds)
                                
{
  const unsigned int teams = width*height/16;
  const unsigned int threads = 4;
  #pragma omp target teams num_teams(teams) thread_limit(threads)
  {
    uchar4 block0[4];
    uchar4 block1[4];
    #pragma omp parallel 
    {

    unsigned int bx = omp_get_team_num() % (width/4);
    unsigned int by = omp_get_team_num() / (width/4);
 
    //unsigned int localIdx = threadIdx.x;
    unsigned int localIdy = omp_get_thread_num();
    
    unsigned int globalIndex = (((by * width/4) + bx) * 4) + (localIdy);
    unsigned int localIndex  = localIdy;

    uchar4 galiosCoeff[4];
    galiosCoeff[0] = {14, 0, 0, 0};
    galiosCoeff[1] = {11, 0, 0, 0};
    galiosCoeff[2] = {13, 0, 0, 0};
    galiosCoeff[3] = { 9, 0, 0, 0};

    block0[localIndex]  = input[globalIndex];
    
    block0[localIndex] ^= roundKey[4*rounds + localIndex];

    for(unsigned int r=rounds -1 ; r > 0; --r)
    {
        block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 
    
        block0[localIndex] = sboxRead(SBox, block0[localIndex]);
        
        #pragma omp barrier
        block1[localIndex] = block0[localIndex]^roundKey[r*4 + localIndex];

        #pragma omp barrier
        block0[localIndex]  = mixColumns(block1, galiosCoeff, localIndex); 
    }  

    block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    output[globalIndex] =  block0[localIndex]^roundKey[localIndex];
    }
  }
}
