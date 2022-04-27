#ifndef __DEFINES__
#define __DEFINES__

#define BLOCK_SIZE2	16
#define BLOCK_SIZE1	128 // have to be power of two

#define SEGMULT 2 // after enforce connectivity the number of segment can increase - we are making a reserve
#define SPXMULT 8 // the size of the superpixels can also increase - we are making reserve

#define MAX3(x,y,z) x > y ? (x > z ? x : z) : (y > z ? y : z)  
#define MIN3(x,y,z) x < y ? (x < z ? x : z) : (y < z ? y : z) 

#endif
