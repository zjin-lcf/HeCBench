void gelu_bias_loop_cpu(sycl::half* src, const sycl::half* bias, int batch_size, int width, int height)
{
  #pragma omp parallel for collapse(3)
  for (int batch = 0; batch < batch_size; batch++) {
    for (int x = 0; x < height; x++) {
      for (int y = 0; y < width; y = y + 2) {
        sycl::half2 v_bias = ((sycl::half2*)bias)[y >> 1];
        sycl::half2 v_src  = ((sycl::half2*)src)[(batch * width * height + x * width + y) >> 1];
        sycl::half2 v      = v_src + v_bias;
        sycl::float2 t = v.convert<float, sycl::rounding_mode::automatic>();
        t.x() = (0.5f * t.x() * (1.0f + sycl::tanh(0.79788456f * (t.x() + 0.044715f * t.x() * t.x() * t.x()))));
        t.y() = (0.5f * t.y() * (1.0f + sycl::tanh(0.79788456f * (t.y() + 0.044715f * t.y() * t.y() * t.y()))));
        ((sycl::half2 *)src)[(batch * width * height + x * width + y) >> 1] = t.convert<sycl::half, sycl::rounding_mode::rte>();
      }
    }
  }
}

