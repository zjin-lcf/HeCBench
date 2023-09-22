#ifndef CU_H
#define CU_H


#ifdef __HIPCC__

#include "hip/hip_runtime.h"
#define CU(x) if ((x) != hipSuccess) { die("%s:%d: HIP error", __FILE__, __LINE__); }
#define DEV __device__
#define HOST __host__
#define GLOBAL __global__

#else

#define DEV
#define HOST
#define GLOBAL

#endif /* __HIPCC__ */

#endif /* CU_H */
