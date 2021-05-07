#pragma once
#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream> // cout
#include <fstream>  // ifstream
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
//--------------------data--------------------//
struct Option {
  std::string inputFile;
  std::string outputFile;
  float threshold;
  int wordLength;
};

struct Read {
  std::string data;
  std::string name;
};

//--------------------function--------------------//
void checkOption(int argc, char **argv, Option &option);
void readFile(std::vector<Read> &reads, Option &option);
