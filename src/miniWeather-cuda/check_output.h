// Reference
// https://github.com/mrnorman/miniWeather/blob/main/cpp/build/check_output.sh

bool check_output(double d_mass, double d_te) {
  if (isnan(d_mass)) {
    printf("Mass change is NaN\n");
    return false;
  }
  if (fabs(d_mass) > 1e-9) {
    printf("Mass change magnitude is too large\n");
    return false;
  }
  if (isnan(d_te)) {
    printf("Total energy change is NaN\n");
    return false;
  }
  if (d_te >= 0) {
    printf("Total energy change must be negative\n");
    return false;
  }
  if (fabs(d_te) > 4.5e-5) {
    printf("Total energy change magnitude is too large\n");
    return false;
  }
  return true;
}
