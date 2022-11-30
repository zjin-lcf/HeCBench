typedef enum {
  ADAM_MODE_0 = 0, // eps under square root
  ADAM_MODE_1 = 1  // eps outside square root
} adamMode_t;

template <typename T, typename G>
void reference (
  int repeat,
  T* p,
  T* m,
  T* v,
  const G* g,
  const float b1,
  const float b2,
  const float eps,
  const float grad_scale,
  const float step_size,
  const size_t tsize,
  adamMode_t mode,
  const float decay)
{
  for (int i = 0; i < repeat; i++) {
    for (size_t j = 0; j < tsize; j++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v[j] + eps);
      else // Mode 1
        denom = sqrtf(v[j]) + eps;
      float update = (m[j]/denom) + (decay*p[j]);
      p[j] = p[j] - (step_size*update);
    }
  }
}

