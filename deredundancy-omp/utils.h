#pragma once
#include <iostream> // cout
#include <fstream>  // ifstream
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
#include <cmath>  // math functions
#include <omp.h>

#define max(a,b) ((a) < (b) ? (b) : (a))
#define min(a,b) ((a) < (b) ? (a) : (b))

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
bool readFile(std::vector<Read> &reads, Option &option);
