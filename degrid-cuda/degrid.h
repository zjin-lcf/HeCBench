#ifndef __DEGRID_H
#define __DEGRID_H

//#define NPOINTS 4000000
#define NPOINTS  400
#define GCF_DIM  256 
#define IMG_SIZE 8192
#define GCF_GRID 8
#define REPEAT   1

// define PRECISION in Makefile
#define PASTER(x) x ## 2
#define EVALUATOR(x) PASTER(x)
#define PRECISION2 EVALUATOR(PRECISION)


#endif
