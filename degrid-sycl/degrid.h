#ifndef __DEGRID_H
#define __DEGRID_H

//#define NPOINTS 4000000
#define NPOINTS  40000
#define GCF_DIM  256 
#define IMG_SIZE 8192
#define GCF_GRID 8
#define REPEAT   100

// define PRECISION in Makefile
#define PASTER(x) x ## 2
#define EVALUATOR(x) PASTER(x)
#define PRECISION2 EVALUATOR(PRECISION)


#endif
