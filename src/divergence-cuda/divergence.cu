
#include <iostream>
#include <fstream>
#include <chrono>

#include "divergence.hpp"

template <int np, typename real>
void readVelocity(real v[np][np][2], std::istream *input) {
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < np; j++) {
      for(int k = 0; k < np; k++) {
        (*input) >> v[k][j][i];
      }
    }
  }
}

template <int np, typename real>
void readElement(element<np, real> &elem,
                 std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> elem.metdet[i][j];
      elem.rmetdet[i][j] = 1 / elem.metdet[i][j];
    }
  }
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      for(int k = 0; k < np; k++) {
        for(int l = 0; l < np; l++) {
          (*input) >> elem.Dinv[l][k][i][j];
        }
      }
    }
  }
}

template <int np, typename real>
void readDerivative(derivative<np, real> &deriv,
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> deriv.Dvv[j][i];
    }
  }
}

template <int np, typename real>
void readDivergence(real divergence[np][np],
                    std::istream *input) {
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      (*input) >> divergence[i][j];
    }
  }
}

constexpr const int DIMS = 2;

template <int np, typename real>
void compareDivergences(const real v[np][np][DIMS],
                        const element<np, real> &elem,
                        const derivative<np, real> &deriv,
                        const real divergence_e[np][np],
                        const int numtests) {
  /* Initial run to prevent cache timing from affecting us
   */
  std::cout << "Divergence on the CPU\n";
  real divergence_c[np][np];
  // warmup
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }

  auto start_c = std::chrono::steady_clock::now();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, real>(v, deriv, elem, divergence_c);
  }
  auto stop_c = std::chrono::steady_clock::now();
  auto time_c = std::chrono::duration_cast<std::chrono::microseconds>(stop_c - start_c).count();

  std::cout << "Divergence on the GPU\n";

  real divergence_f[np][np];
  // warmup
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }

  auto start_f = std::chrono::steady_clock::now();
  for(int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }
  auto stop_f = std::chrono::steady_clock::now();
  auto time_f = std::chrono::duration_cast<std::chrono::microseconds>(stop_f - start_f).count();

  std::cout << "Divergence Errors\n";
  std::cout << "CPU             GPU\n";
  for(int i = 0; i < np; i++) {
    for(int j = 0; j < np; j++) {
      std::cout << divergence_c[i][j] - divergence_e[i][j]
                << "    "
                << divergence_f[i][j] - divergence_e[i][j]
                << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "Total CPU Time: " << time_c * 1e-3f << " (ms)" << std::endl;
  std::cout << "Total GPU Time: " << time_f * 1e-3f << " (ms)" << std::endl;
}

int main(int argc, char **argv) {
  constexpr const int NP = 4;
  real v[NP][NP][DIMS];
  element<NP, real> elem;
  derivative<NP, real> deriv;
  real divergence_e[NP][NP];
  {
    std::istream *input;
    if(argc > 1) {
      input = new std::ifstream(argv[1]);
    } else {
      input = &std::cin;
    }
    readVelocity(v, input);
    readElement(elem, input);
    readDerivative(deriv, input);
    readDivergence(divergence_e, input);
    if(argc > 1) {
      delete input;
    }
  }

  constexpr const int defNumTests = 1e5;
  const int numtests = (argc > 2) ? std::stoi(argv[2]) : defNumTests;
  compareDivergences(v, elem, deriv, divergence_e, numtests);
  return 0;
}
