#pragma once
#include <iostream> // cout
#include <fstream>  // ifstream
#include <vector>  // vector
#include <cstring>  // memcpy
#include <algorithm>  // sort
#include "common.h"
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

/*
struct Data {
  // data form file
  int readsCount;
  int *lengths;
  long *offsets;
  char *reads;
  // data form program
  unsigned int *compressed;
  int *gaps;
  unsigned short *indexs;
  unsigned short *orders;
  long *words;
  int *magicBase;
  // threshold
  int *wordCutoff;
  int *baseCutoff;
};
*/

//--------------------function--------------------//
void checkOption(int argc, char **argv, Option &option);
void readFile(std::vector<Read> &reads, Option &option);
