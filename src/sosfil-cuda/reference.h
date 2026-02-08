#include <vector>

#define sos_width  6   // https://www.mathworks.com/help/signal/ref/sosfilt.html

template <typename T>
void reference(int n_signals, int n_samples, int n_sections, int zi_width,
               const T *sos, // [n_sections * 6]
               const T *zi,  // [n_signals * n_sections * zi_width]
               T *x_in)      // [n_signals * n_samples] (in-place)
{
  // assert(zi_width == 2);

  for (int ty = 0; ty < n_signals; ty++) {

    // Emulate shared memory
    std::vector<T> s_out(n_sections, (T)0);
    std::vector<T> s_zi((n_sections + 1) * n_signals * zi_width, (T)0);
    std::vector<T> s_sos(n_sections * sos_width, (T)0);

    // Load zi
    for (int tx = 0; tx < n_sections; tx++)
      for (int i = 0; i < zi_width; i++)
        s_zi[tx * zi_width + i] =
            zi[ty * n_sections * zi_width + tx * zi_width + i];

    // Load SOS
    for (int tx = 0; tx < n_sections; tx++)
      for (int i = 0; i < sos_width; i++)
        s_sos[tx * sos_width + i] = sos[tx * sos_width + i];

    const int load_size = n_sections - 1;
    const int unload_size = n_samples - load_size;

    T temp;

    // ---------------------------
    // Loading phase
    // ---------------------------
    std::vector<T> x_n(n_sections, 0);
    for (int n = 0; n < load_size; n++) {
      for (int tx = 0; tx < n_sections; tx++) {

        if (tx == 0)
          x_n[tx] = x_in[ty * n_samples + n];
        else
          x_n[tx] = s_out[tx - 1];
      }

      for (int tx = 0; tx < n_sections; tx++) {
        temp = s_sos[tx * sos_width + 0] * x_n[tx] + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] = s_sos[tx * sos_width + 1] * x_n[tx] -
                                  s_sos[tx * sos_width + 4] * temp +
                                  s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n[tx] -
                                  s_sos[tx * sos_width + 5] * temp;

        s_out[tx] = temp;
      }
    }

    // ---------------------------
    // Processing phase
    // ---------------------------
    for (int n = load_size; n < n_samples; n++) {
      for (int tx = 0; tx < n_sections; tx++) {

        if (tx == 0)
          x_n[tx] = x_in[ty * n_samples + n];
        else
          x_n[tx] = s_out[tx - 1];
      }
      for (int tx = 0; tx < n_sections; tx++) {
        temp = s_sos[tx * sos_width + 0] * x_n[tx] + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] = s_sos[tx * sos_width + 1] * x_n[tx] -
                                  s_sos[tx * sos_width + 4] * temp +
                                  s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n[tx] -
                                  s_sos[tx * sos_width + 5] * temp;

        if (tx < load_size)
          s_out[tx] = temp;
        else
          x_in[ty * n_samples + (n - load_size)] = temp;
      }
    }

    // ---------------------------
    // Unloading phase
    // ---------------------------
    for (int n = 0; n < n_sections; n++) {
      for (int tx = n + 1; tx < n_sections; tx++) {

        x_n[tx] = s_out[tx - 1];
      }

      for (int tx = n + 1; tx < n_sections; tx++) {
        temp = s_sos[tx * sos_width + 0] * x_n[tx] + s_zi[tx * zi_width + 0];

        s_zi[tx * zi_width + 0] = s_sos[tx * sos_width + 1] * x_n[tx] -
                                  s_sos[tx * sos_width + 4] * temp +
                                  s_zi[tx * zi_width + 1];

        s_zi[tx * zi_width + 1] = s_sos[tx * sos_width + 2] * x_n[tx] -
                                  s_sos[tx * sos_width + 5] * temp;

        if (tx < load_size)
          s_out[tx] = temp;
        else
          x_in[ty * n_samples + (n + unload_size)] = temp;
      }
    }
  }
}

template <typename T>
bool compare_results(const T *cpu, const T *gpu, int size, T atol, T rtol) {
  for (int i = 0; i < size; i++) {
    T diff = std::abs(cpu[i] - gpu[i]);
    T tol = atol + rtol * std::abs(cpu[i]);

    if (diff > tol) {
      printf("Mismatch at index %d: CPU=%e GPU=%e diff=%e tol=%e\n", i, cpu[i],
             gpu[i], diff, tol);
      return false;
    }
  }
  return true;
}
