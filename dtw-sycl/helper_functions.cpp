/** Kernel to fill a matrix with infinity except for index 0 = 0.0
 *  to initialize the DTW cost matrix
 */

#define max(x,y) (x) > (y) ? (x) : (y)
#define min(x,y) (x) < (y) ? (x) : (y)

void fill_matrix_inf(float *A, unsigned int width, unsigned int height, float val, nd_item<1> &item)
{
    int idx = item.get_global_id(0);
    int gridDim = item.get_group_range(0);
    int blockDim = item.get_local_range(0);

    for (int i = idx; i < width * height; i += gridDim * blockDim)
    {
        A[i] = val;
        if (i % width == 0) A[i] = 0.f;
    }
}
