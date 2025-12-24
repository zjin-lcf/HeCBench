#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

// compute the xyz images using the inverse focal length invF
template<typename T>
void reference (
  const T *s,
  int N,
  T f,
  int w,
  int h,
  T *d)
{
  #pragma omp parallel for collapse(2)
  for( int idy = 0; idy < h; idy++) {
    for( int idx = 0; idx < w; idx++) {
      T ray[3];
      ray[0] = T(idx)-(w-1)*(T)0.5;
      ray[1] = T(idy)-(h-1)*(T)0.5;
      ray[2] = f;
      T pt[3];
      T n[3];
      T p[3];
      T dMin = 1e20;
      
     #pragma omp parallel for reduction(min:dMin)
      for (int i=0; i<N; ++i) {
        p[0] = s[i*COL_DIM+COL_P_X];
        p[1] = s[i*COL_DIM+COL_P_Y];
        p[2] = s[i*COL_DIM+COL_P_Z];
        n[0] = s[i*COL_DIM+COL_N_X];
        n[1] = s[i*COL_DIM+COL_N_Y];
        n[2] = s[i*COL_DIM+COL_N_Z];
        T rSqMax = s[i*COL_DIM+COL_RSq];
        T pDotn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
        T dsDotRay = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2];
        T alpha = pDotn / dsDotRay;
        pt[0] = ray[0]*alpha - p[0];
        pt[1] = ray[1]*alpha - p[1];
        pt[2] = ray[2]*alpha - p[2];
        T t = ray[2]*alpha;
        T rSq = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
        if (rSq < rSqMax && dMin > t) {
          dMin = t; // ray hit the surfel 
        }
      }
      d[idy*w+idx] = dMin > (T)100 ? (T)0 : dMin;
    }
  }
}
