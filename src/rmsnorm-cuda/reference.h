// ----------------------------------------------------------------------------
// CPU code reference

void rmsnorm_forward_cpu(float* out, const float* inp, const float* gamma, int64_t N, int64_t H)
{
  float eps = 1e-5f;
  for (int t = 0; t < N; t++) {
    const float* x = inp + t * H;
    
    // RMS
    float m = 0.0f;
    for (int i = 0; i < H; i++) {
    	m += x[i] * x[i];
    }
    m = m/H;
    float s = 1.0f / sqrtf(m + eps);
    
    float* out_t = out + t * H;
    for (int i = 0; i < H; i++) {
      float o = x[i] * s * gamma[i];
      out_t[i] = o;
    }
  }
}


