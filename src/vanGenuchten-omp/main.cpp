#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

void vanGenuchten(
  const double *__restrict Ksat,
  const double *__restrict psi,
        double *__restrict C,
        double *__restrict theta,
        double *__restrict K,
  const int size)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < size; i++) {

    double Se, _theta, _psi, lambda, m, t;

    lambda = n - 1.0;
    m = lambda/n;

    // Compute the volumetric moisture content [eqn 21]
    _psi = psi[i] * 100.0;
    if ( _psi < 0.0 )
      _theta = (theta_S - theta_R) / pow(1.0 + pow((alpha*(-_psi)),n), m) + theta_R;
    else
      _theta = theta_S;

    theta[i] = _theta;

    // Compute the effective saturation [eqn 2]
    Se = (_theta - theta_R)/(theta_S - theta_R);

    // Compute the hydraulic conductivity [eqn 8]
    t = 1.0 - pow(1.0-pow(Se,1.0/m), m);
    K[i] = Ksat[i] * sqrt(Se) * t * t;

    // Compute the specific moisture storage derivative of eqn (21).
    // So we have to calculate C = d(theta)/dh. Then the unit is converted into [1/m].
    if (_psi < 0.0)
      C[i] = 100 * alpha * n * (1.0/n-1.0)*pow(alpha*abs(_psi), n-1.0)
        * (theta_R-theta_S) * pow(pow(alpha*abs(_psi), n)+1.0, 1.0/n-2.0);
    else
      C[i] = 0.0;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: ./%s <dimX> <dimY> <dimZ> <repeat>\n", argv[0]);
    return 1;
  }

  const int dimX = atoi(argv[1]);
  const int dimY = atoi(argv[2]);
  const int dimZ = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int size = dimX * dimY * dimZ;

  double *Ksat, *psi, *C, *theta, *K;
  double *C_ref, *theta_ref, *K_ref;
  
  Ksat = new double[size];
  psi = new double[size];
  C = new double[size];
  theta = new double[size];
  K = new double[size];

  C_ref = new double[size];
  theta_ref = new double[size];
  K_ref = new double[size];

  // arbitrary numbers
  for (int i = 0; i < size; i++) {
    Ksat[i] = 1e-6 +  (1.0 - 1e-6) * i / size; 
    psi[i] = -100.0 + 101.0 * i / size;
  }

  // for verification
  reference(Ksat, psi, C_ref, theta_ref, K_ref, size);

  #pragma omp target data map(to: Ksat[0:size], psi[0:size]) \
                          map(from: C[0:size], theta[0:size], K[0:size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      vanGenuchten(Ksat, psi, C, theta, K, size);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (fabs(C[i] - C_ref[i]) > 1e-3 || 
        fabs(theta[i] - theta_ref[i]) > 1e-3 ||
        fabs(K[i] - K_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete(Ksat);
  delete(psi);
  delete(C);
  delete(theta);
  delete(K);
  delete(C_ref);
  delete(theta_ref);
  delete(K_ref);

  return ok ? 0 : 1;
}
