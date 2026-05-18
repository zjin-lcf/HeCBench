#include <algorithm>

void stencil(
    int iteration,
    const float* power,
    const float* temp_src,
    float* temp_dst,
    int grid_cols,
    int grid_rows,
    float Cap,
    float Rx,
    float Ry,
    float Rz,
    float step)
{
  const float amb_temp = 80.0f;
  const float step_div_Cap = step / Cap;
  const float Rx_1 = 1.0f / Rx;
  const float Ry_1 = 1.0f / Ry;
  const float Rz_1 = 1.0f / Rz;

  const int total_size = grid_rows * grid_cols;

  float *curr = new float[total_size];
  float *next = new float[total_size];

  memcpy(curr, temp_src, total_size * sizeof(float));

  for (int iter = 0; iter < iteration; iter++) {
    for (int y = 0; y < grid_rows; y++) {
      for (int x = 0; x < grid_cols; x++) {
        const int idx = y * grid_cols + x;

        //  boundaries
        const int N = std::max(y - 1, 0);
        const int S = std::min(y + 1, grid_rows - 1);
        const int W = std::max(x - 1, 0);
        const int E = std::min(x + 1, grid_cols - 1);

        const float temp = curr[idx];
        const float tempN = curr[N * grid_cols + x];
        const float tempS = curr[S * grid_cols + x];
        const float tempW = curr[y * grid_cols + W];
        const float tempE = curr[y * grid_cols + E];

        next[idx] = temp + step_div_Cap * ( power[idx] +
                    ( tempS + tempN - 2.f * temp) * Ry_1 +
                    ( tempE + tempW - 2.f * temp) * Rx_1 +
                    ( amb_temp - temp) * Rz_1 );
      }
    }
    std::swap(curr, next);
  }

  memcpy(temp_dst, curr, total_size * sizeof(float));
  delete[] curr;
  delete[] next;
}

int reference(
    const float *MatrixPower,
          float *MatrixTemp[2],
    int col, int row,
    int total_iterations, int num_iterations)
{
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  int t;
#ifdef DEBUG
  printf("%f %f %f %f %f %f %f\n", grid_height,grid_width,Cap,Rx,Ry,Rz,step);
#endif

  int src = 0, dst = 1;

  for (t = 0; t < total_iterations; t += num_iterations) {

    int iter = MIN(num_iterations, total_iterations - t);

    stencil(iter, MatrixPower, MatrixTemp[src], MatrixTemp[dst], col, row, Cap, Rx, Ry, Rz, step);

    src = 1 - src;
    dst = 1 - dst;
  }
  return src;
}
