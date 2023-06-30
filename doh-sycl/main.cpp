#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

typedef float IMAGE_T;
typedef int INT_T;

/* @func _clip
    Clip coordinate between low and high values.
    Parameters
    ----------
    x : int
        Coordinate to be clipped.
    low : int
        The lower bound.
    high : int
        The higher bound.
    Returns
    -------
    x : int
        `x` clipped between `high` and `low`.
    """
    assert 0 <= low <= high

    if x > high:
        return high
    elif x < low:
        return low
    else:
        return x
*/

inline int _clip(const int x, const int low, const int high)
{
  if (x > high)
    return high;
  else if (x < low)
    return low;
  else
    return x;
}


/*@func _integ
    Integrate over the 2D integral image in the given window.
    Parameters
    ----------
    img : array
        The integral image over which to integrate.
    r : int
        The row number of the top left corner.
    c : int
        The column number of the top left corner.
    rl : int
        The number of rows over which to integrate.
    cl : int
        The number of columns over which to integrate.
    Returns
    -------
    ans : double
        The integral over the given window.
*/

inline IMAGE_T _integ(const IMAGE_T * img,
                      const INT_T img_rows,
                      const INT_T img_cols,
                      int r,
                      int c,
                      const int rl,
                      const int cl)
{
  r = _clip(r, 0, img_rows - 1);
  c = _clip(c, 0, img_cols - 1);

  const int r2 = _clip(r + rl, 0, img_rows - 1);
  const int c2 = _clip(c + cl, 0, img_cols - 1);

  IMAGE_T ans = img[r * img_cols + c] + img[r2 * img_cols + c2] -
                img[r * img_cols + c2] - img[r2 * img_cols + c];

  return sycl::max((IMAGE_T)0, ans);
}


/*@func hessian_matrix_det
    Compute the approximate Hessian Determinant over a 2D image.
    This method uses box filters over integral images to compute the
    approximate Hessian Determinant as described in [1]_.
    Parameters
    ----------
    img : array
        The integral image over which to compute Hessian Determinant.
    sigma : double
        Standard deviation used for the Gaussian kernel, used for the Hessian
        matrix
    Returns
    -------
    out : array
        The array of the Determinant of Hessians.
    References
    ----------
    .. [1] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf
    Notes
    -----
    The running time of this method only depends on size of the image. It is
    independent of `sigma` as one would expect. The downside is that the
    result for `sigma` less than `3` is not accurate, i.e., not similar to
    the result obtained if someone computed the Hessian and took its
    determinant.

    This function is derived from Scikit-Image _hessian_matrix_det (v0.19.3):
*/

void hessian_matrix_det(const IMAGE_T* img,
                        const INT_T img_rows,
                        const INT_T img_cols,
                        const IMAGE_T sigma,
                        IMAGE_T* out,
                        sycl::nd_item<1> item)
{
  int tid = item.get_global_id(0);

  if (tid >= img_rows*img_cols) return;

  const int r = tid / img_cols;
  const int c = tid % img_cols;

  int size = (int)((IMAGE_T)3.0 * sigma);

  const int b = (size - 1) / 2 + 1;
  const int l = size / 3;
  const int w = size;

  const IMAGE_T w_i = (IMAGE_T)1.0 / (size * size);

  const IMAGE_T tl = _integ(img, img_rows, img_cols, r - l, c - l, l, l); // top left
  const IMAGE_T br = _integ(img, img_rows, img_cols, r + 1, c + 1, l, l); // bottom right
  const IMAGE_T bl = _integ(img, img_rows, img_cols, r - l, c + 1, l, l); // bottom left
  const IMAGE_T tr = _integ(img, img_rows, img_cols, r + 1, c - l, l, l); // top right

  IMAGE_T dxy = bl + tr - tl - br;
  dxy = -dxy * w_i;

  IMAGE_T mid = _integ(img, img_rows, img_cols, r - l + 1, c - l, 2 * l - 1, w);  // middle box
  IMAGE_T side = _integ(img, img_rows, img_cols, r - l + 1, c - l / 2, 2 * l - 1, l);  // sides

  IMAGE_T dxx = mid - (IMAGE_T)3 * side;
  dxx = -dxx * w_i;

  mid = _integ(img, img_rows, img_cols, r - l, c - b + 1, w, 2 * b - 1);
  side = _integ(img, img_rows, img_cols, r - b / 2, c - b + 1, b, 2 * b - 1);

  IMAGE_T dyy = mid - (IMAGE_T)3 * side;
  dyy = -dyy * w_i;

  out[tid] = (dxx * dyy - (IMAGE_T)0.81 * (dxy * dxy));
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <height> <width> <repeat>\n", argv[0]);
    return 1;
  }

  int h = atoi(argv[1]);
  int w = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  size_t img_size = h * w;
  size_t img_size_bytes = sizeof(float) * img_size;

  float *input_img = (float*) malloc (img_size_bytes);
  float *integral_img = (float*) malloc (img_size_bytes);
  float *output_img = (float*) malloc (img_size_bytes);

  std::default_random_engine rng (123);
  std::normal_distribution<float> norm_dist(0.f, 1.f);

  for (size_t i = 0; i < img_size; i++) {
    input_img[i] = norm_dist(rng);
  }

  printf("Integrating the input image may take a while...\n"); 
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      IMAGE_T s = 0;
      for (int y = 0; y <= i; y++)
        for (int x = 0; x <= j; x++)
          s += input_img[y * w + x];
      integral_img[i * w + j] = s;
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_input_img = sycl::malloc_device<float>(img_size, q);

  q.memcpy(d_input_img, integral_img, img_size_bytes);

  float *d_output_img = sycl::malloc_device<float>(img_size, q);

  const IMAGE_T sigma = 4.0;

  sycl::range<1> gws ((img_size + 255) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        hessian_matrix_det(d_input_img, h, w, sigma, d_output_img, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(output_img, d_output_img, img_size_bytes).wait();

  double checksum = 0;
  for (size_t i = 0; i < img_size; i++) {
    checksum += output_img[i];
  }

  sycl::free(d_input_img, q);
  sycl::free(d_output_img, q);
  free(input_img);
  free(integral_img);
  free(output_img);

  printf("Average kernel execution time : %f (us)\n", time * 1e-3 / repeat);
  printf("Kernel checksum: %lf\n", checksum);
  return 0;
}
