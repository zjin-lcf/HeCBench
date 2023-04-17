// called by host and device
unsigned char
galoisMultiplication(unsigned char a, unsigned char b)
{
    unsigned char p = 0; 
    for(unsigned int i=0; i < 8; ++i)
    {
        if((b&1) == 1)
        {
            p^=a;
        }
        unsigned char hiBitSet = (a & 0x80);
        a <<= 1;
        if(hiBitSet == 0x80)
        {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    return p;
}

inline uchar4
sboxRead(const uchar * SBox, uchar4 block)
{
    return {SBox[block.x()], SBox[block.y()], SBox[block.z()], SBox[block.w()]};
}

uchar4
mixColumns(const uchar4 * block, const uchar4 * galiosCoeff, unsigned int j)
{
    unsigned int bw = 4;

    uchar x, y, z, w;

    x = galoisMultiplication(block[0].x(), galiosCoeff[(bw-j)%bw].x());
    y = galoisMultiplication(block[0].y(), galiosCoeff[(bw-j)%bw].x());
    z = galoisMultiplication(block[0].z(), galiosCoeff[(bw-j)%bw].x());
    w = galoisMultiplication(block[0].w(), galiosCoeff[(bw-j)%bw].x());
   
    for(unsigned int k=1; k< 4; ++k)
    {
        x ^= galoisMultiplication(block[k].x(), galiosCoeff[(k+bw-j)%bw].x());
        y ^= galoisMultiplication(block[k].y(), galiosCoeff[(k+bw-j)%bw].x());
        z ^= galoisMultiplication(block[k].z(), galiosCoeff[(k+bw-j)%bw].x());
        w ^= galoisMultiplication(block[k].w(), galiosCoeff[(k+bw-j)%bw].x());
    }
    
    return {x, y, z, w};
}

uchar4
shiftRows(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        //r.xyzw() = r.yzwx();
        uchar x = r.x();
        uchar y = r.y();
        uchar z = r.z();
        uchar w = r.w();
        r = {y,z,w,x};
    }
    return r;
}

void AESEncrypt(      uchar4  *__restrict output  ,
                const uchar4  *__restrict input   ,
                const uchar4  *__restrict roundKey,
                const uchar   *__restrict SBox    ,
                      uchar4  *__restrict block0  ,  // lmem
                      uchar4  *__restrict block1  ,  // lmem
                const uint     width , 
                const uint     rounds,
                sycl::nd_item<2>     item   )
                                
{
    unsigned int blockIdx = item.get_group(1);
    unsigned int blockIdy = item.get_group(0);
 
    //unsigned int localIdx = item.get_local_id(1);
    unsigned int localIdy = item.get_local_id(0);
    
    unsigned int globalIndex = (((blockIdy * width/4) + blockIdx) * 4 )+ (localIdy);
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
       
        item.barrier(sycl::access::fence_space::local_space);
        block1[localIndex]  = mixColumns(block0, galiosCoeff, localIndex); 
        
        item.barrier(sycl::access::fence_space::local_space);
        block0[localIndex] = block1[localIndex]^roundKey[r*4 + localIndex];
    }  
    block0[localIndex] = sboxRead(SBox, block0[localIndex]);
  
    block0[localIndex] = shiftRows(block0[localIndex], localIndex); 

    output[globalIndex] =  block0[localIndex]^roundKey[(rounds)*4 + localIndex];
}

uchar4
shiftRowsInv(uchar4 row, unsigned int j)
{
    uchar4 r = row;
    for(uint i=0; i < j; ++i)  
    {
        // r = r.wxyz();
        uchar x = r.x();
        uchar y = r.y();
        uchar z = r.z();
        uchar w = r.w();
        r = {w,x,y,z};
    }
    return r;
}

void AESDecrypt(       uchar4  *__restrict output    ,
                const  uchar4  *__restrict input     ,
                const  uchar4  *__restrict roundKey  ,
                const  uchar   *__restrict SBox      ,
                       uchar4  *__restrict block0    ,
                       uchar4  *__restrict block1    ,
                const  uint    width , 
                const  uint    rounds,
                sycl::nd_item<2>     item   )
                                
{
    unsigned int blockIdx = item.get_group(1);
    unsigned int blockIdy = item.get_group(0);
 
    //unsigned int localIdx = item.get_local_id(1);
    unsigned int localIdy = item.get_local_id(0);
    
    unsigned int globalIndex = (((blockIdy * width/4) + blockIdx) * 4 )+ (localIdy);
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
        
        item.barrier(sycl::access::fence_space::local_space);
        block1[localIndex] = block0[localIndex]^roundKey[r*4 + localIndex];

        item.barrier(sycl::access::fence_space::local_space);
        block0[localIndex]  = mixColumns(block1, galiosCoeff, localIndex); 
    }  

    block0[localIndex] = shiftRowsInv(block0[localIndex], localIndex); 

    block0[localIndex] = sboxRead(SBox, block0[localIndex]);

    output[globalIndex] =  block0[localIndex]^roundKey[localIndex];
}
