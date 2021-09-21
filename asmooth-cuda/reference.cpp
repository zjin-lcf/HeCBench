void reference (int Lx, int Ly, int threshold, int maxRad, 
                float *img, int *box, float *norm, float *out)
{
  float q, sum, s;
  int ksum;

  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      sum = 0.f;
      s = q = 1;  // box size
      ksum = 0; // kernel sum

      while (sum < threshold && q < maxRad) {
        s = q;
        sum = 0.f;
        ksum = 0;

        for (int i = -s; i < s+1; i++)
          for (int j = -s; j < s+1; j++)
            if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
              sum += img[(x+i)*Ly+y+j];
              ksum++;
            }
        q++;
      }

      box[x*Ly+y] = s;  // save the box size

      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++)
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly)
            if (ksum != 0) norm[(x+i)*Ly+y+j] += 1.f / (float)ksum;
    }
  }

  // normalize the image
  for (int x = 0; x < Lx; x++)
    for (int y = 0; y < Ly; y++) 
      if (norm[x*Ly+y] != 0) img[x*Ly+y] /= norm[x*Ly+y];

  // output file
  for (int x = 0; x < Lx; x++) {
    for (int y = 0; y < Ly; y++) {
      s = box[x*Ly+y];
      sum = 0.f;
      ksum = 0;

      // resmooth with normalized image
      for (int i = -s; i < s+1; i++)
        for (int j = -s; j < s+1; j++) {
          if (x-s >=0 && x+s < Lx && y-s >=0 && y+s < Ly) {
            sum += img[(x+i)*Ly+y+j];
            ksum++;
          }
        }
      if (ksum != 0) out[x*Ly+y] = sum / (float)ksum;
    }
  }
}
