#ifndef _NEARESTNEIGHBOR
#define _NEARESTNEIGHBOR

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>

// All OpenCL headers

#include <algorithm>


	
#define REC_LENGTH 49 // size of a record in db

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

void FindNearestNeighbors(
	int numRecords,
	std::vector<LatLong> &locations,float lat,float lng,
	float* distances,
	int timing);

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t);
#endif
