#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <vector>

#include "heap.h"

#define WARPSIZE 32

#ifndef BX
#define BX  16
#endif

#ifndef BY
#define BY  8
#endif

#ifndef BZ
#define BZ  4
#endif

#ifndef BLOCKSIZE1D 
#define BLOCKSIZE1D 256
#endif


void printarray(float *a, int n, const char *name);

void printarrayfrag(float *a, int m, const char *name);

void printindexarrayfrag(float *a, findex* ind, int m, const char *name);

void reset_array(float *a, int n, float val);

void fgoleft(findex_t *frag, int ar);

findex_t fgetleft(findex_t frag, int ar);

void newtemp(float *aT, int *ar, int *R, findex_t l);

void rebuild_temps(float *aT, int R, int ar);

void insert_temps(float *aavex, float *aT, int *R, int *ar, int ains);

void rebuild_indices(findex_t* arts, findex_t *atrs, int ar) ;

double rtclock();

#endif
