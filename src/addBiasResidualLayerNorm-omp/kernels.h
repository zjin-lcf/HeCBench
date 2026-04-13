/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

template<typename T>
inline float typeToFloat(T a) {
  return float(a);
}

template<typename T>
inline T floatToType(float a) {
  return T(a);
};

template<typename T>
inline T add(T a, T b) {
  return a + b;
}

template<typename T>
inline T add(T a, T b, T c) {
  return a + b + c;
}

template<typename T>
inline T sub(T a, T b) {
  return a - b;
}

template<typename T>
inline T fma(T a, T b, T c, T d) {
    return a * b * c + d;
}

template<typename T>
void generalAddBiasResidualPostLayerNorm_omp(
          T*__restrict__ out,
    const T*__restrict__ input,
    const T*__restrict__ bias,
    const T*__restrict__ gamma,
    const T*__restrict__ beta,
    const float layernorm_eps,
    const int m,   // number of rows (batch)
    const int n,   // hidden size
    const int block_size)
{
  #pragma omp target teams distribute num_teams(m)
  for (int row = 0; row < m; row++) {

    float mean = 0.0f;

    #pragma omp parallel for reduction(+:mean) num_threads(block_size)
    for (int col = 0; col < n; col++) {
      int idx = row * n + col;

      float val = typeToFloat(add(out[idx], input[idx], bias[col]));

      out[idx] = floatToType<T>(val);
      mean += val;
    }

    mean /= n;

    float variance = 0.0f;
    #pragma omp parallel for reduction(+:variance) num_threads(block_size)
    for (int col = 0; col < n; col++) {
      int idx = row * n + col;

      float val = typeToFloat(out[idx]);
      float diff = val - mean;
      variance += diff * diff;
    }

    float inv_std = 1.f / sqrtf(variance / n + layernorm_eps);

    #pragma omp parallel for num_threads(block_size)
    for (int col = 0; col < n; col++) {
      int idx = row * n + col;

      float val = typeToFloat(out[idx]);

      float norm = (val - mean) * inv_std * typeToFloat(gamma[col]) +
                   typeToFloat(beta[col]);

      out[idx] = floatToType<T>(norm);
    }
  }
}
