//bondsEngine.cu
//Scott Grauer-Gray sgrauerg@gmail.com
//Contains main function for running bonds application on a GPU

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 
#include "bondsStructs.h"
#include "bondsKernelsGpu.cu"
#include "bondsKernelsCpu.cu"

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))


int monthLengthCpu(int month, bool leapYear) 
{
  int MonthLength[] = {
    31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };

  int MonthLeapLength[] = {
    31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
  };

  return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}


int monthOffsetCpu(int m, bool leapYear) 
{
  int MonthOffset[] = {
    0,  31,  59,  90, 120, 151,   // Jan - Jun
    181, 212, 243, 273, 304, 334,   // Jun - Dec
    365     // used in dayOfMonth to bracket day
  };

  int MonthLeapOffset[] = {
    0,  31,  60,  91, 121, 152,   // Jan - Jun
    182, 213, 244, 274, 305, 335,   // Jun - Dec
    366     // used in dayOfMonth to bracket day
  };

  return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}


int yearOffsetCpu(int y)
{
  // the list of all December 31st in the preceding year
  // e.g. for 1901 yearOffset[1] is 366, that is, December 31 1900
  int YearOffset[] = {
    // 1900-1909
    0,  366,  731, 1096, 1461, 1827, 2192, 2557, 2922, 3288,
    // 1910-1919
    3653, 4018, 4383, 4749, 5114, 5479, 5844, 6210, 6575, 6940,
    // 1920-1929
    7305, 7671, 8036, 8401, 8766, 9132, 9497, 9862,10227,10593,
    // 1930-1939
    10958,11323,11688,12054,12419,12784,13149,13515,13880,14245,
    // 1940-1949
    14610,14976,15341,15706,16071,16437,16802,17167,17532,17898,
    // 1950-1959
    18263,18628,18993,19359,19724,20089,20454,20820,21185,21550,
    // 1960-1969
    21915,22281,22646,23011,23376,23742,24107,24472,24837,25203,
    // 1970-1979
    25568,25933,26298,26664,27029,27394,27759,28125,28490,28855,
    // 1980-1989
    29220,29586,29951,30316,30681,31047,31412,31777,32142,32508,
    // 1990-1999
    32873,33238,33603,33969,34334,34699,35064,35430,35795,36160,
    // 2000-2009
    36525,36891,37256,37621,37986,38352,38717,39082,39447,39813,
    // 2010-2019
    40178,40543,40908,41274,41639,42004,42369,42735,43100,43465,
    // 2020-2029
    43830,44196,44561,44926,45291,45657,46022,46387,46752,47118,
    // 2030-2039
    47483,47848,48213,48579,48944,49309,49674,50040,50405,50770,
    // 2040-2049
    51135,51501,51866,52231,52596,52962,53327,53692,54057,54423,
    // 2050-2059
    54788,55153,55518,55884,56249,56614,56979,57345,57710,58075,
    // 2060-2069
    58440,58806,59171,59536,59901,60267,60632,60997,61362,61728,
    // 2070-2079
    62093,62458,62823,63189,63554,63919,64284,64650,65015,65380,
    // 2080-2089
    65745,66111,66476,66841,67206,67572,67937,68302,68667,69033,
    // 2090-2099
    69398,69763,70128,70494,70859,71224,71589,71955,72320,72685,
    // 2100-2109
    73050,73415,73780,74145,74510,74876,75241,75606,75971,76337,
    // 2110-2119
    76702,77067,77432,77798,78163,78528,78893,79259,79624,79989,
    // 2120-2129
    80354,80720,81085,81450,81815,82181,82546,82911,83276,83642,
    // 2130-2139
    84007,84372,84737,85103,85468,85833,86198,86564,86929,87294,
    // 2140-2149
    87659,88025,88390,88755,89120,89486,89851,90216,90581,90947,
    // 2150-2159
    91312,91677,92042,92408,92773,93138,93503,93869,94234,94599,
    // 2160-2169
    94964,95330,95695,96060,96425,96791,97156,97521,97886,98252,
    // 2170-2179
    98617,98982,99347,99713,100078,100443,100808,101174,101539,101904,
    // 2180-2189
    102269,102635,103000,103365,103730,104096,104461,104826,105191,105557,
    // 2190-2199
    105922,106287,106652,107018,107383,107748,108113,108479,108844,109209,
    // 2200
    109574
  };

  return YearOffset[y-1900];
}


bool isLeapCpu(int y) 
{
  bool YearIsLeap[] = {
    // 1900 is leap in agreement with Excel's bug
    // 1900 is out of valid date range anyway
    // 1900-1909
    true,false,false,false, true,false,false,false, true,false,
    // 1910-1919
    false,false, true,false,false,false, true,false,false,false,
    // 1920-1929
    true,false,false,false, true,false,false,false, true,false,
    // 1930-1939
    false,false, true,false,false,false, true,false,false,false,
    // 1940-1949
    true,false,false,false, true,false,false,false, true,false,
    // 1950-1959
    false,false, true,false,false,false, true,false,false,false,
    // 1960-1969
    true,false,false,false, true,false,false,false, true,false,
    // 1970-1979
    false,false, true,false,false,false, true,false,false,false,
    // 1980-1989
    true,false,false,false, true,false,false,false, true,false,
    // 1990-1999
    false,false, true,false,false,false, true,false,false,false,
    // 2000-2009
    true,false,false,false, true,false,false,false, true,false,
    // 2010-2019
    false,false, true,false,false,false, true,false,false,false,
    // 2020-2029
    true,false,false,false, true,false,false,false, true,false,
    // 2030-2039
    false,false, true,false,false,false, true,false,false,false,
    // 2040-2049
    true,false,false,false, true,false,false,false, true,false,
    // 2050-2059
    false,false, true,false,false,false, true,false,false,false,
    // 2060-2069
    true,false,false,false, true,false,false,false, true,false,
    // 2070-2079
    false,false, true,false,false,false, true,false,false,false,
    // 2080-2089
    true,false,false,false, true,false,false,false, true,false,
    // 2090-2099
    false,false, true,false,false,false, true,false,false,false,
    // 2100-2109
    false,false,false,false, true,false,false,false, true,false,
    // 2110-2119
    false,false, true,false,false,false, true,false,false,false,
    // 2120-2129
    true,false,false,false, true,false,false,false, true,false,
    // 2130-2139
    false,false, true,false,false,false, true,false,false,false,
    // 2140-2149
    true,false,false,false, true,false,false,false, true,false,
    // 2150-2159
    false,false, true,false,false,false, true,false,false,false,
    // 2160-2169
    true,false,false,false, true,false,false,false, true,false,
    // 2170-2179
    false,false, true,false,false,false, true,false,false,false,
    // 2180-2189
    true,false,false,false, true,false,false,false, true,false,
    // 2190-2199
    false,false, true,false,false,false, true,false,false,false,
    // 2200
    false
  };

  return YearIsLeap[y-1900];
}


bondsDateStruct intializeDateCpu(int d, int m, int y) 
{
  bondsDateStruct currDate;

  currDate.day = d;
  currDate.month = m;
  currDate.year = y;

  bool leap = isLeapCpu(y);
  int offset = monthOffsetCpu(m,leap);

  currDate.dateSerialNum = d + offset + yearOffsetCpu(y);

  return currDate;
}

void runBoundsEngine(const int repeat)
{
  //can run multiple times with different number of bonds by uncommenting these lines
  int nBondsArray[] = {1000000};

  for (int numTime=0; numTime < 1; numTime++)
  {
    int numBonds = nBondsArray[numTime];  
    printf("\nNumber of Bonds: %d\n\n", numBonds);

    inArgsStruct inArgsHost;
    inArgsHost.discountCurve = (bondsYieldTermStruct*)malloc(numBonds*sizeof(bondsYieldTermStruct));
    inArgsHost.repoCurve = (bondsYieldTermStruct*)malloc(numBonds*sizeof(bondsYieldTermStruct));
    inArgsHost.currDate = (bondsDateStruct*)malloc(numBonds*sizeof(bondsDateStruct));
    inArgsHost.maturityDate = (bondsDateStruct*)malloc(numBonds*sizeof(bondsDateStruct));
    inArgsHost.bondCleanPrice = (dataType*)malloc(numBonds*sizeof(dataType));
    inArgsHost.bond = (bondStruct*)malloc(numBonds*sizeof(bondStruct));
    inArgsHost.dummyStrike = (dataType*)malloc(numBonds*sizeof(dataType));

    srand (123);

    int numBond;
    for (numBond = 0; numBond < numBonds; numBond++)
    {
      dataType repoRate = 0.07;

      //int repoSettlementDays = 0;
      int repoCompounding = SIMPLE_INTEREST;
      dataType repoCompoundFreq = 1;

      // assume a ten year bond- this is irrelevant
      bondsDateStruct bondIssueDate =  intializeDateCpu(rand() % 28 + 1, rand() % 12 + 1, 1999 - (rand() % 2));
      bondsDateStruct bondMaturityDate = intializeDateCpu(rand() % 28 + 1, rand() % 12 + 1, 2000 + (rand() % 2));

      bondsDateStruct todaysDate = intializeDateCpu(bondMaturityDate.day-1,bondMaturityDate.month,bondMaturityDate.year);

      bondStruct bond;
      bond.startDate = bondIssueDate;
      bond.maturityDate = bondMaturityDate;
      bond.rate = 0.08 + ((float)rand()/(float)RAND_MAX - 0.5)*0.1;

      dataType bondCouponFrequency = 2;

      dataType bondCleanPrice = 89.97693786;

      bondsYieldTermStruct bondCurve;

      bondCurve.refDate = todaysDate;
      bondCurve.calDate = todaysDate;
      bondCurve.forward = -0.1f;  // dummy rate
      bondCurve.compounding = COMPOUNDED_INTEREST;
      bondCurve.frequency = bondCouponFrequency;
      bondCurve.dayCounter = USE_EXACT_DAY;

      bondCurve.refDate = todaysDate;
      bondCurve.calDate = todaysDate;
      bondCurve.compounding = COMPOUNDED_INTEREST;
      bondCurve.frequency = bondCouponFrequency;

      dataType dummyStrike = 91.5745;

      bondsYieldTermStruct repoCurve;
      repoCurve.refDate = todaysDate;
      repoCurve.calDate = todaysDate;
      repoCurve.forward = repoRate;
      repoCurve.compounding = repoCompounding;
      repoCurve.frequency = repoCompoundFreq;
      repoCurve.dayCounter = USE_SERIAL_NUMS;

      inArgsHost.discountCurve[numBond] = bondCurve;
      inArgsHost.repoCurve[numBond] = repoCurve;
      inArgsHost.currDate[numBond] = todaysDate;
      inArgsHost.maturityDate[numBond] = bondMaturityDate;
      inArgsHost.bondCleanPrice[numBond] = bondCleanPrice;
      inArgsHost.bond[numBond] = bond;
      inArgsHost.dummyStrike[numBond] = dummyStrike;
    }
    printf("Inputs for bond with index %d\n", numBonds/2);
    printf("Bond Issue Date: %d-%d-%d\n", inArgsHost.bond[numBonds/2].startDate.month, 
                                          inArgsHost.bond[numBonds/2].startDate.day, 
                                          inArgsHost.bond[numBonds/2].startDate.year);
    printf("Bond Maturity Date: %d-%d-%d\n", inArgsHost.bond[numBonds/2].maturityDate.month, 
                                          inArgsHost.bond[numBonds/2].maturityDate.day, 
                                          inArgsHost.bond[numBonds/2].maturityDate.year);
    printf("Bond rate: %f\n\n", inArgsHost.bond[numBonds/2].rate);

    resultsStruct resultsHost;
    resultsStruct resultsFromGpu;

    resultsHost.dirtyPrice = (dataType*)malloc(numBonds*sizeof(dataType));
    resultsHost.accruedAmountCurrDate = (dataType*)malloc(numBonds*sizeof(dataType));;
    resultsHost.cleanPrice = (dataType*)malloc(numBonds*sizeof(dataType));;
    resultsHost.bondForwardVal = (dataType*)malloc(numBonds*sizeof(dataType));;

    resultsFromGpu.dirtyPrice = (dataType*)malloc(numBonds*sizeof(dataType));
    resultsFromGpu.accruedAmountCurrDate = (dataType*)malloc(numBonds*sizeof(dataType));;
    resultsFromGpu.cleanPrice = (dataType*)malloc(numBonds*sizeof(dataType));;
    resultsFromGpu.bondForwardVal = (dataType*)malloc(numBonds*sizeof(dataType));;

    long ktimeGpu = 0;
    double timeCpu;
    double timeGpu;

    struct timeval start;
    struct timeval end;

    gettimeofday(&start, NULL);

    for (int i = 0; i < repeat; i++)
      ktimeGpu += getBondsResultsGpu(inArgsHost, resultsFromGpu, numBonds);

    gettimeofday(&end, NULL);
    timeGpu = (end.tv_sec - start.tv_sec) * 1e6 + end.tv_usec - start.tv_usec;

    printf("Run on GPU\n");
    printf("Average kernel execution time on GPU: %lf (ms)  \n\n", ktimeGpu * 1e-3 / repeat);
    printf("Average processing time on GPU: %f (ms)  \n\n", timeGpu * 1e-3 / repeat);

    double totPrice = 0.0;
    int numBond1;
    for (numBond1= 0; numBond1< numBonds; numBond1++)
    {
      totPrice += resultsFromGpu.dirtyPrice[numBond1];
    }

    printf("Sum of output dirty prices on GPU: %f\n", totPrice);
    printf("Outputs on GPU for bond with index %d: \n", numBonds/2);
    printf("Dirty Price: %f\n", resultsFromGpu.dirtyPrice[numBonds/2]);
    printf("Accrued Amount: %f\n", resultsFromGpu.accruedAmountCurrDate[numBonds/2]);
    printf("Clean Price: %f\n", resultsFromGpu.cleanPrice[numBonds/2]);
    printf("Bond Forward Val: %f\n\n", resultsFromGpu.bondForwardVal[numBonds/2]);

    gettimeofday(&start, NULL);

    for (int i = 0; i < 2; i++)
      getBondsResultsCpu(inArgsHost, resultsHost, numBonds);

    gettimeofday(&end, NULL);
    timeCpu = (end.tv_sec - start.tv_sec) * 1e6 + end.tv_usec - start.tv_usec;

    printf("Run on CPU\n");
    printf("Average processing time on CPU: %lf (ms)  \n\n", timeCpu * 1e-3 / 2);

    totPrice = 0.0;
    for (numBond1= 0; numBond1< numBonds; numBond1++)
    {
      totPrice += resultsHost.dirtyPrice[numBond1];
    }
    printf("Sum of output dirty prices on CPU: %f\n", totPrice);
    printf("Outputs on CPU for bond with index %d: \n", numBonds/2);
    printf("Dirty Price: %f\n", resultsHost.dirtyPrice[numBonds/2]);
    printf("Accrued Amount: %f\n", resultsHost.accruedAmountCurrDate[numBonds/2]);
    printf("Clean Price: %f\n", resultsHost.cleanPrice[numBonds/2]);
    printf("Bond Forward Val: %f\n\n", resultsHost.bondForwardVal[numBonds/2]);

    printf("Speedup using GPU: %f\n", (timeCpu / 2) / (timeGpu / repeat) );

    free(resultsHost.dirtyPrice);
    free(resultsHost.accruedAmountCurrDate);;
    free(resultsHost.cleanPrice);;
    free(resultsHost.bondForwardVal);;

    free(resultsFromGpu.dirtyPrice);
    free(resultsFromGpu.accruedAmountCurrDate);;
    free(resultsFromGpu.cleanPrice);;
    free(resultsFromGpu.bondForwardVal);

    free(inArgsHost.discountCurve);
    free(inArgsHost.repoCurve);
    free(inArgsHost.currDate);
    free(inArgsHost.maturityDate);
    free(inArgsHost.bondCleanPrice);
    free(inArgsHost.bond);
    free(inArgsHost.dummyStrike);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  runBoundsEngine(repeat);
  return 0;
}
