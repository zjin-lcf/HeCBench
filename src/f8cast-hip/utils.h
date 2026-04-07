void init (bool isE4M3, float *src, int nelems) {
  std::mt19937 gen(19937);

  float min = isE4M3 ? -240 : -57344;
  float max = isE4M3 ?  240 :  57344;
  std::uniform_real_distribution<float> dis(min, max+1); 

  // specific values
  if (isE4M3) {
    src[0] = powf(2.f, -10);
    src[1] = 0.875 * src[0];
    src[2] = powf(2.f, -11);
    src[3] = 240;
    src[4] = 448;
  }
  else {
    src[0] = powf(2.f, -17); // 4
    src[1] = 0.75 * src[0]; // 3
    src[2] = powf(2.f, -18); // 1
    src[3] = 57344; // 7B
    src[4] = 57345;
  }
  src[5] = INFINITY;
  src[6] = -INFINITY;
  src[7] = NAN;
  for (int i = 8; i < nelems; i++) {
    src[i] = dis(gen); 
  }
}
