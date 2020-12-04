#include <sys/time.h>
#include <cstdio>
#include <map>
#include <string>

using namespace std;

struct Timer {

  map<string, float> startTime;
  map<string, float> stopTime;
  map<string, float> time;

  void start(string name) {
    if(!time.count(name)) {
      time[name] = 0.0;
    }
    startTime[name] = clock();
  }

  void stop(string name) {
    stopTime[name] = clock();
    float part_time = stopTime[name] - startTime[name];
    time[name] += part_time;
  }

  void print(string name, unsigned int REP = 1) { 
    printf("%s Time (s): %.3f\n", name.c_str(), time[name] / CLOCKS_PER_SEC / REP); 
  }

  void release(string name){
  }
};
