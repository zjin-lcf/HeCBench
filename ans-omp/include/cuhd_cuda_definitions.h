/*****************************************************************************
 *
 * MULTIANS - Massively parallel ANS decoding on GPUs
 *
 * released under LGPL-3.0
 *
 * 2017-2019 André Weißenberger
 *
 *****************************************************************************/

#ifndef CUHD_OMP_DEFINITIONS_
#define CUHD_OMP_DEFINITIONS_

#include <omp.h>

#include <iostream>

typedef struct { std::uint32_t x; std::uint32_t y; std::uint32_t z; std::uint32_t w; } uint4 ;

#endif /* CUHD_OMP_DEFINITIONS */
