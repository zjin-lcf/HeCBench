void silu_forward_reference(
    const float* x,
    float* out,
    uint64_t N)
{
  #pragma omp parallel for
  for (uint64_t idx = 0; idx < N; idx++) {
    float x_val = x[idx];
    out[idx] = x_val / (1.0f + expf(-x_val));
  }
}

void silu_backward_reference(
    const float* dout, const float* x,
    float* dx,
    uint64_t N)
{
  #pragma omp parallel for
  for (uint64_t idx = 0; idx < N; idx++) {
    float out_val = dout[idx];
    float x_val = x[idx];
    float expx = expf(-x_val);
    float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
    dx[idx] = out_val * grad_silu;
  }
}
