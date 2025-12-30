void init (bool isE4M3, float *src, int nelems) {
  std::mt19937 gen(19937);

  float min = isE4M3 ? -448 : -57344;
  float max = isE4M3 ?  449 :  57345;
  std::uniform_real_distribution<float> dis(min, max); 

  // specific values
  if (isE4M3) {
    src[0] = powf(2.f, -6); // 8
    src[1] = 0.875 * src[0]; // 7
    src[2] = powf(2.f, -9); // 1
    src[3] = 448; // 7E
    src[4] = 449;
  }
  else {
    src[0] = powf(2.f, -14); // 4
    src[1] = 0.75 * src[0]; // 3
    src[2] = powf(2.f, -16); // 1
    src[3] = 57344; // 7B
    src[4] = 57345;
  }
  for (int i = 5; i < nelems; i++) {
    src[i] = dis(gen); 
  }
}

