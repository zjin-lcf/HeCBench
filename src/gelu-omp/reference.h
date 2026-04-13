void gelu_bias_loop_cpu(_Float16* src, const _Float16* bias, int batch_size, int width, int height)
{
  #pragma omp parallel for collapse(3)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int x = 0; x < height; x++) {
      for (int y = 0; y < width; y++) {
        auto v_bias = bias[y];
        auto v_src  = src[(batch * width * height + x * width + y)];
        auto t      = (float)(v_src + v_bias);
        t    = (0.5f * t * (1.0f + tanhf(0.79788456f * (t + 0.044715f * t * t * t))));
        src[(batch * width * height + x * width + y)] = _Float16(t);
      }
    }
  }
}

