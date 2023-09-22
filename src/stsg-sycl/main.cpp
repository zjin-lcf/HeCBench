//******************************************************************
//cuSTSG is used to reconstruct high-quality NDVI time series data(MODIS/SPOT) based on STSG
//
//This procedure cuSTSG is the source code for the first version of cuSTSG.
//This is a parallel computing code using GPU.
//
//Coded by Yang Xue
// Reference:Xue Yang, Jin Chen, Qingfeng Guan, Huan Gao, and Wei Xia.
// Enhanced Spatial-Temporal Savitzky-Golay Method for Reconstructing High-quality NDVI Time Series: Reduced Sensitivity
// to Quality Flags and Improved Computational Efficiency.Transactions on Geoscience and Remote Sensing
//******************************************************************

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <gdal/gdal_priv.h>
#include <sycl/sycl.hpp>
#include "Filter.h"

using namespace std;

int main(int argc, char *argv[])
{
  GDALAllRegister();

  //parameters
  if (argc != 2)
  {
    cout << "No parameter file!" << endl;
    return 1;
  }

  ifstream parameter(argv[1]);
  if (!parameter)
  {
    cout << "Can't open parameter file!" << endl;
    return 1;
  }

  int* Years = nullptr;
  string NDVI_path, Reliability_path, STSG_Test_path;
  float cosyear, sampcorr;
  int win_year, win, snow_address, n_Years;
  string par;
  while (getline(parameter, par))
  {
    if (par.substr(0, 2) == "//" || par == "")
      continue;

    for (int i = 0; i < par.size(); )
    {
      if (isspace(par[i]))
        par.erase(i,1);
      else
        i++;
    }
    if (par.substr(0, par.find("=")) == "Years")
    {
      vector<int> year;
      while (par.rfind(",") < par.size())
      {
        year.push_back(stoi(par.substr(par.rfind(",") + 1)));
        par = par.substr(0, par.rfind(","));
      }
      year.push_back(stoi(par.substr(par.rfind("=") + 1)));

      n_Years = year.size();
      Years = new int[n_Years];
      for (int i = 0; i < n_Years; i++)
        Years[i] = year[n_Years - i - 1];
    }
    else if (par.substr(0, par.find("=")) == "NDVI_path")
      NDVI_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "Reliability_path")
      Reliability_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "STSG_Test_path")
      STSG_Test_path = par.substr(par.find("=") + 1);
    else if (par.substr(0, par.find("=")) == "cosyear")
      cosyear = stof(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "win_year")
      win_year = stoi(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "win")
      win = stoi(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "sampcorr")
      sampcorr = stof(par.substr(par.find("=") + 1));
    else if (par.substr(0, par.find("=")) == "snow_address")
      snow_address = stoi(par.substr(par.find("=") + 1));
  }
  parameter.close();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  cout << "Device name:" << q.get_device().get_info<sycl::info::device::name>() << endl;
  size_t gmem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
  size_t const totalGlobalMem = std::min(gmem_size, 4UL * 1024 * 1024 * 1024);
  cout << "Device global memory used: " << totalGlobalMem / 1024 / 1024 << " MB" << endl;

  vector<GDALDataset*> NDVI(n_Years);
  vector<GDALDataset*> QA(n_Years);
  int n_X, n_Y, n_B;
  GDALDataType type_NDVI, type_QA;
  for (int i = 0; i < n_Years; i++)
  {
    string FileName = NDVI_path + to_string(Years[i]);
    NDVI[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
    if (i == 0)
    {
      n_X = NDVI[i]->GetRasterXSize();
      n_Y = NDVI[i]->GetRasterYSize();
      n_B = NDVI[i]->GetRasterCount();
      type_NDVI = NDVI[i]->GetRasterBand(1)->GetRasterDataType();
    }

    FileName = Reliability_path + to_string(Years[i]);
    QA[i] = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
    if (i == 0)
      type_QA = QA[i]->GetRasterBand(1)->GetRasterDataType();
  }

  cout << "Execution start " << endl;
  size_t PerYSize = n_X*n_B *(n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(int) + sizeof(float) + n_Years * sizeof(float)) + n_X*(2 * win + 1)*(2 * win + 1) *(7 * sizeof(float) + 3 * sizeof(int));
  if (totalGlobalMem <= 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) + n_X*n_Y*n_B*n_Years*sizeof(float))
  {
    cout << "Size of vector_out is larger than total device global memory. Exit!" << endl;
    return 1;
  }

  size_t PerStep = (totalGlobalMem - 2 * win*n_X*n_B* (n_Years * sizeof(short) + n_Years * sizeof(unsigned char) + 2 * n_Years * sizeof(float) + sizeof(float)) - n_X*n_Y*n_B*n_Years*sizeof(float)) / PerYSize;
  int Loops = 1;
  if (PerStep < n_Y)
  {
    Loops = n_Y / PerStep + 1;
    PerStep = n_Y / Loops + 1;
  }

  float *d_vector_out;
  size_t nBytes = n_X*n_Y*n_B*n_Years * sizeof(float);
  d_vector_out = (float *)sycl::malloc_device(nBytes, q);
  q.memset((void *)d_vector_out, 0, nBytes).wait();
  nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
  float *res = (float*)malloc(nBytes);
  memset((void*)res, 0, nBytes);
  int last_Buffer_Dn = 0;

  printf("Number of loops: %d\n", Loops);
  for (int i = 1, StartY = 0; i <= Loops && StartY < n_Y; i++, StartY += PerStep)
  {
    cout << "Loops " << i << endl;
    if (i == Loops)
      PerStep = n_Y - StartY;

    int Buffer_Up = 0;
    int Buffer_Dn = 0;
    if (StartY + PerStep < n_Y - win)
      Buffer_Dn = win;
    else
      Buffer_Dn = n_Y - PerStep - StartY;
    if (StartY >= win)
      Buffer_Up = win;
    else
      Buffer_Up = StartY;

    int blkwidth = 16;
    int blkheight = 16;
    sycl::range<2> blocks(blkheight, blkwidth);
    sycl::range<2> grids(
        (PerStep + Buffer_Up + Buffer_Dn) % blkheight == 0
            ? (PerStep + Buffer_Up + Buffer_Dn) / blkheight
            : (PerStep + Buffer_Up + Buffer_Dn) / blkheight + 1,
        n_X % blkwidth == 0 ? n_X / blkwidth : n_X / blkwidth + 1);

    short *img_NDVI = new short[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
    unsigned char *img_QA = new unsigned char[(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years];
    for (int i = 0; i < n_Years; i++)
    {
      NDVI[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_NDVI[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_NDVI, n_B, nullptr, 0, 0, 0);
      QA[i]->RasterIO(GF_Read, 0, StartY - Buffer_Up, n_X, (PerStep + Buffer_Up + Buffer_Dn), &img_QA[i*(PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B], n_X, (PerStep + Buffer_Up + Buffer_Dn), type_QA, n_B, nullptr, 0, 0, 0);
    }

    short *d_imgNDVI;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(short);
    d_imgNDVI = (short *)sycl::malloc_device(nBytes, q);
    q.memcpy((void *)d_imgNDVI, (void *)img_NDVI, nBytes);
    unsigned char *d_imgQA;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(unsigned char);
    d_imgQA = (unsigned char *)sycl::malloc_device(nBytes, q);
    q.memcpy((void *)d_imgQA, (void *)img_QA, nBytes);
    float *d_img_NDVI, *d_img_QA;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B*n_Years * sizeof(float);
    d_img_NDVI = (float *)sycl::malloc_device(nBytes, q);
    d_img_QA = (float *)sycl::malloc_device(nBytes, q);
    q.memset((void *)d_img_NDVI, 0, nBytes);
    q.memset((void *)d_img_QA, 0, nBytes);
    float *d_NDVI_Reference, *d_res;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B * sizeof(float);
    d_NDVI_Reference = (float *)sycl::malloc_device(nBytes, q);
    q.memset((void *)d_NDVI_Reference, 0, nBytes);
    nBytes = (PerStep + Buffer_Dn)*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    d_res = (float *)sycl::malloc_device(nBytes, q);
    q.memset((void *)d_res, 0, nBytes);
    nBytes = last_Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    q.memcpy((void *)d_res, (void *)res, nBytes);

    int *d_res_vec_res1;
    nBytes = (PerStep + Buffer_Up + Buffer_Dn)*n_X*n_B * sizeof(int);
    d_res_vec_res1 = (int *)sycl::malloc_device(nBytes, q);
    q.memset((void *)d_res_vec_res1, 0, nBytes);
    float *d_vector_in, *d_res_3;
    nBytes = PerStep*n_X* n_B * sizeof(float);
    d_vector_in = (float *)sycl::malloc_device(nBytes, q);
    q.memset((void *)d_vector_in, 0, nBytes);
    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
    d_res_3 = (float *)sycl::malloc_device(nBytes, q); //(slope_intercept(2);corr_similar;)
    q.memset((void *)d_res_3, 0, nBytes);
    int *d_index;
    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(int);
    d_index = (int *)sycl::malloc_device(nBytes, q); //(similar_index(2);new_corr;)
    q.memset(d_index, 0, nBytes);
    q.wait();

    sycl::range<2> gws (grids * blocks);
    sycl::range<2> lws (blocks);

    auto start = std::chrono::steady_clock::now();
    q.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
      Short_to_Float(d_imgNDVI, d_imgQA, n_X,
                     (PerStep + Buffer_Up + Buffer_Dn), n_B,
                     n_Years, d_img_NDVI, d_img_QA,
                     item);
    });

    q.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
      Generate_NDVI_reference(cosyear, win_year, d_img_NDVI, d_img_QA, n_X,
                              (PerStep + Buffer_Up + Buffer_Dn), n_B, n_Years,
                              d_NDVI_Reference, d_res_3, d_res_vec_res1,
                              item);
    });

    nBytes = PerStep*n_X*(2 * win + 1)*(2 * win + 1) * 3 * sizeof(float);
    q.memset((void *)d_res_3, 0, nBytes);

    q.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
      Compute_d_res(d_img_NDVI, d_img_QA, d_NDVI_Reference,
                    StartY, n_Y, Buffer_Up, Buffer_Dn, n_X,
                    (PerStep + Buffer_Up + Buffer_Dn), n_B,
                    n_Years, win, d_res, item);
    });

    q.parallel_for(sycl::nd_range<2>(gws, lws), [=](sycl::nd_item<2> item) {
      STSG_filter(d_img_NDVI, d_img_QA, d_NDVI_Reference,
                  StartY, n_Y, Buffer_Up, Buffer_Dn, n_X,
                  PerStep, n_B, n_Years, win, sampcorr,
                  snow_address, d_vector_out, d_vector_in,
                  d_res, d_res_3, d_index, item);
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cout << "Total kernel time: " << time * 1e-9f  << " s" << endl;

    nBytes = win*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    memset((void*)res, 0, nBytes);
    nBytes = Buffer_Dn*n_X*(2 * win + 1)*(2 * win + 1) * 4 * sizeof(float);
    q.memcpy(res, &d_res[(PerStep + Buffer_Dn - win) * n_X *
                         (2 * win + 1) * (2 * win + 1) * 4], nBytes).wait();
    last_Buffer_Dn = Buffer_Dn;

    delete[] img_NDVI;
    delete[] img_QA;
    sycl::free((void *)d_imgNDVI, q);
    sycl::free((void *)d_imgQA, q);
    sycl::free((void *)d_img_NDVI, q);
    sycl::free((void *)d_img_QA, q);
    sycl::free((void *)d_NDVI_Reference, q);
    sycl::free((void *)d_res, q);
    sycl::free((void *)d_res_vec_res1, q);
    sycl::free((void *)d_vector_in, q);
    sycl::free((void *)d_res_3, q);
    sycl::free((void *)d_index, q);
  }
  free((void*)res);

  float *vector_out = new float[n_X*n_Y*n_B*n_Years];
  nBytes = n_X*n_Y*n_B*n_Years* sizeof(float);
  q.memcpy((void *)vector_out, (void *)d_vector_out, nBytes).wait();
  sycl::free((void *)d_vector_out, q);

  long cnt = 0;
  double sum = 0;
  for (int i = 0; i < n_X*n_Y*n_B*n_Years; i++) {
    if (vector_out[i] < 1.f || vector_out[i] > 0.f) {
      sum += vector_out[i];
      cnt++;
    }
  }
  cout << "Checksum: " << sum << " " << cnt << " " << sum / cnt << endl;

  delete[] vector_out;

  return 0;
}
