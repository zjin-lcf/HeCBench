#include <chrono>
#include <sycl/sycl.hpp>
#include "nearestNeighbor.h"

int main(int argc, char *argv[]) {
  std::vector<Record> records;
  float *recordDistances;
  std::vector<LatLong> locations;
  int i;
  char filename[100];
  int resultsCount=10,quiet=0,timing=0;
  int repeat=1;
  float lat=0.0,lng=0.0;

  if (parseCommandline(argc, argv, filename, &resultsCount,
                       &lat, &lng, &repeat, &quiet, &timing)) {
    printUsage();
    return 0;
  }

  int numRecords = loadData(filename,records,locations);

  if (!quiet) {
    printf("Number of records: %d\n",numRecords);
    printf("Finding the %d closest neighbors.\n",resultsCount);
  }

  if (resultsCount > numRecords) resultsCount = numRecords;

  auto start = std::chrono::steady_clock::now();

  recordDistances = (float *)malloc(sizeof(float) * numRecords);
  SyclFindNearestNeighbors(numRecords,locations,lat,lng,recordDistances,repeat,timing);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing)
    printf("Device offloading time %f (s)\n", time * 1e-9);

  // find the resultsCount least distances
  findLowest(records,recordDistances,numRecords,resultsCount);

  // print out results
  if (!quiet)
    for(i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    }
  free(recordDistances);
  return 0;
}

void SyclFindNearestNeighbors(
    int numRecords,
    std::vector<LatLong> &locations,
    float lat,
    float lng,
    float* distances,
    int repeat,
    int timing) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  LatLong* d_locations = sycl::malloc_device<LatLong>(numRecords, q);
  q.memcpy(d_locations, locations.data(), sizeof(LatLong) * numRecords);

  float *d_distances = sycl::malloc_device<float>(numRecords, q);

  size_t localWorkSize  = 64;
  size_t globalWorkSize = (numRecords + localWorkSize-1) / localWorkSize * localWorkSize;
  sycl::range<1> gws (globalWorkSize);
  sycl::range<1> lws (localWorkSize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  // measure the total kernel execution time
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class nn>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        if (gid < numRecords) {
          LatLong latLong = d_locations[gid];
          d_distances[gid] = sycl::sqrt(
           (lat-latLong.lat)*(lat-latLong.lat)+(lng-latLong.lng)*(lng-latLong.lng));
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(distances, d_distances, numRecords * sizeof(float)).wait();
  sycl::free(d_locations, q);
  sycl::free(d_distances, q);
}

int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
  FILE   *flist,*fp;
  int    i=0;
  char dbname[64];
  int recNum=0;

  /**Main processing **/

  flist = fopen(filename, "r");
  while(!feof(flist)) {
    /**
     * Read in REC_WINDOW records of length REC_LENGTH
     * If this is the last file in the filelist, then done
     * else open next file to be read next iteration
     */
    if(fscanf(flist, "%s\n", dbname) != 1) {
      fprintf(stderr, "error reading filelist\n");
      exit(0);
    }
    fp = fopen(dbname, "r");
    if(!fp) {
      printf("error opening a db\n");
      exit(1);
    }
    // read each record
    while(!feof(fp)){
      Record record;
      LatLong latLong;
      fgets(record.recString,49,fp);
      fgetc(fp); // newline
      if (feof(fp)) break;

      // parse for lat and long
      char substr[6];

      for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
      substr[5] = '\0';
      latLong.lat = atof(substr);

      for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
      substr[5] = '\0';
      latLong.lng = atof(substr);

      locations.push_back(latLong);
      records.push_back(record);
      recNum++;
    }
    fclose(fp);
  }
  fclose(flist);
  return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,
                     int *r, float *lat, float *lng, int *repeat, int *q, int *t) {
  int i;
  if (argc < 2) return 1; // error
  strncpy(filename,argv[1],100);
  char flag;

  for(i=1;i<argc;i++) {
    if (argv[i][0]=='-') {// flag
      flag = argv[i][1];
      switch (flag) {
        case 'r': // number of results
          i++;
          *r = atoi(argv[i]);
          break;
        case 'l': // lat or lng
          if (argv[i][2]=='a') {//lat
            *lat = atof(argv[i+1]);
          }
          else {//lng
            *lng = atof(argv[i+1]);
          }
          i++;
          break;
        case 'i': // i
          *repeat = atoi(argv[i+1]);
          i++;
          break;
        case 'h': // help
          return 1;
          break;
        case 'q': // quiet
          *q = 1;
          break;
        case 't': // timing
          *t = 1;
          break;
      }
    }
  }
  return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] \n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90 -i 100\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-i [int]     kernel execution count (default: 1)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
}
