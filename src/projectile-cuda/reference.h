const float kPIValue = 3.1415;
const float kGValue = 9.81;

void reference (const Projectile *obj, Projectile *pObj, const int num_elements) {
  #pragma omp parallel for simd
  for (int i = 0; i < num_elements; i++) {
    float proj_angle = obj[i].getangle();
    float proj_vel = obj[i].getvelocity();
    float sin_value = sinf(proj_angle * kPIValue / 180.0f);
    float cos_value = cosf(proj_angle * kPIValue / 180.0f);
    float total_time = fabsf((2 * proj_vel * sin_value)) / kGValue;
    float max_range =  fabsf(proj_vel * total_time * cos_value);
    float max_height = (proj_vel * proj_vel * sin_value * sin_value) / 2.0f *
                       kGValue;  // h = v^2 * sin^2theta/2g
    pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel, max_height);
  }
}

