//blackScholesAnalyticEngine.cpp
//Functions for running black scholes using the analytic engine (from Quantlib) on the GPU

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <sycl/sycl.hpp>

#define NUM_DIFF_SETTINGS 37

//needed for optionInputStruct
#include "blackScholesAnalyticEngineStructs.h"

//needed for the kernel(s) to run on the GPU
#include "blackScholesAnalyticEngineKernels.cpp"

#include "blackScholesAnalyticEngineKernelsCpu.cpp"

//function to run the black scholes analytic engine on the gpu
void runBlackScholesAnalyticEngine(const int repeat)
{
  int numberOfSamples = 50000000;
  {
    int numVals = numberOfSamples;//nSamplesArray[numTime];

    optionInputStruct* values = new optionInputStruct[numVals];

    for (int numOption = 0; numOption < numVals; numOption++)
    {
      if ((numOption % NUM_DIFF_SETTINGS) == 0)
      {
        optionInputStruct currVal = { CALL,  40.00,  42.00, 0.08, 0.04, 0.75, 0.35,  5.0975, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 1)
      {
        optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  0.0205, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 2)
      {
        optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 3)
      {
        optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  9.9413, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 4)
      {
        optionInputStruct currVal = { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.25,  0.3150, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 5)
      {
        optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 6)
      {
        optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.25, 10.3556, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 7)
      {
        optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.10, 0.35,  0.9474, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 8)
      {
        optionInputStruct currVal = { CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 9)
      {
        optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.35, 11.1381, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 10)
      {
        optionInputStruct currVal =  { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.15,  0.8069, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 11)
      {
        optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 12)
      {
        optionInputStruct currVal =  { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.15, 10.5769, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 13)
      {
        optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.25,  2.7026, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 14)
      {
        optionInputStruct currVal =   { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 15)
      {
        optionInputStruct currVal =   { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.25, 12.7857, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 16)
      {
        optionInputStruct currVal =   { CALL, 100.00,  90.00, 0.10, 0.10, 0.50, 0.35,  4.9329, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 17)
      {
        optionInputStruct currVal =  { CALL, 100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 18)
      {
        optionInputStruct currVal = { CALL, 100.00, 110.00, 0.10, 0.10, 0.50, 0.35, 15.3086, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 19)
      {
        optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.15,  9.9210, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 20)
      {
        optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.15,  1.8734, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 21)
      {
        optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.15,  0.0408, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 22)
      {
        optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.25, 10.2155, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 23)
      {
        optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.25,  3.1217, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 24)
      {
        optionInputStruct currVal =    { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.25,  0.4551, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 25)
      {
        optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.10, 0.35, 10.8479, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 26)
      {
        optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.10, 0.35,  4.3693, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 27)
      {
        optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.10, 0.35,  1.2376, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 28)
      {
        optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.15, 10.3192, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 29)
      {
        optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.15,  4.0232, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 30)
      {
        optionInputStruct currVal =  { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.15,  1.0646, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 31)
      {
        optionInputStruct currVal =  { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.25, 12.2149, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 32)
      {
        optionInputStruct currVal =   { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.25,  6.6997, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 33)
      {
        optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.25,  3.2734, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 34)
      {
        optionInputStruct currVal =   { PUT,  100.00,  90.00, 0.10, 0.10, 0.50, 0.35, 14.4452, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 35)
      {
        optionInputStruct currVal =  { PUT,  100.00, 100.00, 0.10, 0.10, 0.50, 0.35,  9.3679, 1.0e-4};
        values[numOption] = currVal;
      }
      if ((numOption % NUM_DIFF_SETTINGS) == 36)
      {
        optionInputStruct currVal =   { PUT,  100.00, 110.00, 0.10, 0.10, 0.50, 0.35,  5.7963, 1.0e-4};
        values[numOption] = currVal;
      }
    }

    // Run GPU code

    //initialize the arrays

    //declare and allocate the input and output data on the CPU
    float* outputVals = (float*)malloc(numVals * sizeof(float));

    printf("Number of options: %d\n\n", numVals);
    long seconds, useconds, kseconds, kuseconds;
    float mtimeCpu, mtimeGpu, ktimeGpu;
    struct timeval start;
    gettimeofday(&start, NULL);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    //allocate space for data on GPU
    optionInputStruct* optionsGpu = sycl::malloc_device<optionInputStruct>(numVals, q);
    float* outputValsGpu = sycl::malloc_device<float>(numVals, q);

    //copy the data from the CPU to the GPU
    q.memcpy(optionsGpu, values, numVals * sizeof(optionInputStruct)).wait();

    // setup execution parameters
    sycl::range<1> gws ((numVals + THREAD_BLOCK_SIZE - 1) / 
                        THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE);
    sycl::range<1> lws (THREAD_BLOCK_SIZE);

    struct timeval kstart;
    gettimeofday(&kstart, NULL);

    for (int i = 0; i < repeat; i++) {
      q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<class blackScholesKernel>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int optionNum = item.get_global_id(0);

          //check if within current options
          if (optionNum < numVals)
          {
            const optionInputStruct threadOption = optionsGpu[optionNum];

            payoffStruct currPayoff;
            currPayoff.type = threadOption.type;
            currPayoff.strike = threadOption.strike;

            yieldTermStruct qTS;
            qTS.timeYearFraction = threadOption.t;
            qTS.forward = threadOption.q;

            yieldTermStruct rTS;
            rTS.timeYearFraction = threadOption.t;
            rTS.forward = threadOption.r;

            blackVolStruct volTS;
            volTS.timeYearFraction = threadOption.t;
            volTS.volatility = threadOption.vol;

            blackScholesMertStruct stochProcess;
            stochProcess.x0 = threadOption.spot;
            stochProcess.dividendTS = qTS;
            stochProcess.riskFreeTS = rTS;
            stochProcess.blackVolTS = volTS;

            optionStruct currOption;
            currOption.payoff = currPayoff;
            currOption.yearFractionTime = threadOption.t;
            currOption.pricingEngine = stochProcess; 

            float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
            float dividendDiscount = getDiscountOnDividendYield(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
            float riskFreeDiscount = getDiscountOnRiskFreeRate(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
            float spot = currOption.pricingEngine.x0; 

            float forwardPrice = spot * dividendDiscount / riskFreeDiscount;

            //declare the blackCalcStruct
            blackCalcStruct blackCalc;

            //initialize the calculator
            initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sycl::sqrt(variance), riskFreeDiscount);

            //retrieve the results values
            float resultVal = getResultVal(blackCalc);

            //write the resulting value to global memory
            outputValsGpu[optionNum] = resultVal;
          }
        });
      });
    }
    q.wait();

    struct timeval kend;
    gettimeofday(&kend, NULL);
    kseconds  = kend.tv_sec  - kstart.tv_sec;
    kuseconds = kend.tv_usec - kstart.tv_usec;
    ktimeGpu = ((kseconds) * 1000 + ((float)kuseconds)/1000.0) + 0.5f;

    q.memcpy(outputVals, outputValsGpu, numVals * sizeof(float)).wait();

    sycl::free(optionsGpu, q);
    sycl::free(outputValsGpu, q);

    struct timeval end;
    gettimeofday(&end, NULL);

    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;

    mtimeGpu = ((seconds) * 1000 + ((float)useconds)/1000.0) + 0.5f;

    printf("Run on GPU\n");
    printf("Average kernel execution time on GPU: %f (ms)\n", ktimeGpu / repeat);

    mtimeGpu -= ktimeGpu + ktimeGpu / repeat;
    printf("Processing time on GPU: %f (ms)\n", mtimeGpu);

    float totResult = 0.0f;
    for (int i=0; i<numVals; i++)
    {
      totResult += outputVals[i];
    }
    printf("Summation of output prices on GPU: %f\n", totResult);
    printf("Output price at index %d on GPU: %f\n\n", numVals/2, outputVals[numVals/2]);

    //run on CPU
    gettimeofday(&start, NULL);
    for (int numOption=0; numOption < numVals; numOption++)
    {
      getOutValOptionCpu(values, outputVals, numOption, numVals);  
    }
    gettimeofday(&end, NULL);
    seconds  = end.tv_sec  - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    mtimeCpu = ((seconds) * 1000 + ((float)useconds)/1000.0) + 0.5f;
    printf("Run on CPU\n");
    printf("Processing time on CPU: %f (ms)\n", mtimeCpu);
    totResult = 0.0f;
    for (int i=0; i<numVals; i++)
    {
      totResult += outputVals[i];
    }

    printf("Summation of output prices on CPU: %f\n", totResult);
    printf("Output price at index %d on CPU: %f\n\n", numVals/2, outputVals[numVals/2]);

    printf("Speedup on GPU: %f\n", mtimeCpu / mtimeGpu);

    delete [] values;
    free(outputVals);
  }
}

int main( int argc, char** argv) 
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  runBlackScholesAnalyticEngine(repeat);
  return 0;
}
