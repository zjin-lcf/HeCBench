#include "utils.h"
#include "kernels.h"

//-----------------------------------------------------------------------------------
// MAIN PROCESSING FUNCTION
//-----------------------------------------------------------------------------------

/*
 * Process a given query
 */
int processQuery(queue &q,
                 const std::vector<std::string> &refFiles, 
                 const std::vector<std::string> &sigGeneNameList,
                 const std::vector<int> &sigRegValue,
                 const int nRandomGenerations,
                 const int compoundChoice,
                 const int ENFPvalue, std::ofstream &outputStream) 
{
  // The total number of genes is fixed in this implementation (although the code may be modified)
  const int nGenesTotal = U133AArrayLength;

  // Unordered Cmax calculation score
  int sigNGenes = sigGeneNameList.size();
  double UCmax = computeUCMax(sigNGenes, nGenesTotal);

  // Create a vector in which to store the reference gene name list
  std::vector<std::string> refGeneNameList;
  populateRefGeneNameList(refFiles.front(), refGeneNameList);
  // Create an array on the host side, initialized with zero elements
  int qIndex[nGenesTotal] = {0}; // The array size is known (and generally not too big), so can be static...
  // Otherwise would need to use 'int *qIndex = new int[refLines];' and later 'delete [] qIndex;'
  int errorFlag = queryToIndex(qIndex, sigGeneNameList, sigRegValue, refGeneNameList);
  // Check for an error - break out of the loop if one is found (don't just return - we need to ensure device memory is freed)
  if (errorFlag != 0) {
    std::cout << "Error finding all required genes" << std::endl;
    return -1;
  }

  // Preallocate arrays for random numbers & signature additions on the GPU
  int signatureByRNGs = nRandomGenerations * sigNGenes;

  // Create and seed a random number generator to use later
  // Create an array of uniformly distributed random numbers between 0 and 1, to be used to help obtain p-values
  std::default_random_engine generator (123);
  std::uniform_real_distribution<float> distribution(0.f, 1.f);
  float *randomValues = (float*) malloc(sizeof(float) * signatureByRNGs);

  buffer<float, 1> device_randomValues (signatureByRNGs);
  buffer<float, 1> device_arraysAdded (nRandomGenerations);

  // Preallocate a host array to read in the up/down regulation values (-1/1) for the file currently being read
  int refRegValues[nGenesTotal];

  // Preallocate device arrays
  buffer<int, 1> device_refRegValues (nGenesTotal); // Main device array corresponding to refRegValues

  // Device array to contain up/down regulation values for genes in the query, and zeros where the gene was not in the query
  buffer<int, 1> device_queryRefRegValues (qIndex, nGenesTotal);

  // Initialize the set size to zero
  int setSize = 0;

  // We're going to start processing the query... add a line to the output by way of introduction
  outputStream << "The results against the reference profiles are listed below" << std::endl;
  outputStream << "Compound" << "\t" << "setSize" << "\t" << "averageSetScore" << "\t" << "P-value result"<<"\t " << "ENFP"<< "\t" << "Significant"<< std::endl;

  // Loop through all the ref files in turn
  for (int refFileLoop = 0; refFileLoop < refFiles.size(); refFileLoop++) {

    // Display percentage completed
    if(refFileLoop % 1500 == 0) {
      std::cout << "Completed : " << (int)(((float)refFileLoop / refFiles.size()) * 100) << "%" << std::endl;
    }

    // Read the next input file & increment the set size
    populateRefRegulationValues(refFiles[refFileLoop], refRegValues, setSize > 0);
    setSize++;

    // Get a string for this drug
    std::string drug = parseDrugInfoFromFilename(refFiles[refFileLoop], compoundChoice);

    // If we aren't processing the last file, check whether the next file name produces the same drug string -
    // if so, we'll have a set size > 1.
    // We need to combine the information from multiply files before we can do any further processing
    if ((refFileLoop < refFiles.size() - 1) && (drug == parseDrugInfoFromFilename(refFiles[refFileLoop+1], compoundChoice)))
      continue;

    // We only get here when we're done accumulating info for a particular drug

    // Compute the set score for the signature, and normalize it to UCmax
    int blocksPerGrid = (int)ceil((float)nGenesTotal / (float)threadsPerBlock);

    q.submit([&] (handler &cgh) {
      auto acc = device_refRegValues.get_access<sycl_write>(cgh);
      cgh.copy(refRegValues, acc);
    });

    double cumulativeSetScore = computeDotProduct(q, device_refRegValues, device_queryRefRegValues, 
                                nGenesTotal, blocksPerGrid, threadsPerBlock) / UCmax;

    // If we have multiple sets involved, divide by the set size
    double averageSetScore = cumulativeSetScore / setSize;

    // generate random values
    for (int i = 0; i < signatureByRNGs; ++i) randomValues[i] = distribution(generator);
    q.submit([&] (handler &cgh) {
      auto acc = device_randomValues.get_access<sycl_write>(cgh);
      cgh.copy(randomValues, acc);
    });

    // Compute a p-value using the random numbers to create random gene signatures, and compare these to the reference profile
    double pValue = computePValue(q, nRandomGenerations, threadsPerBlock, averageSetScore, setSize, signatureByRNGs, UCmax,
        device_randomValues, device_refRegValues, device_arraysAdded);

    // The total number of drugs will depend upon the compound choice selection -
    // determine this and assess significance
    int nDrugs = getNDrugs(compoundChoice);
    double ENFP = pValue * nDrugs;
    int significant = ENFP < ENFPvalue;

    // Write the results to the output stream
    outputStream << drug << "\t" << setSize << "\t" << averageSetScore << "\t" << pValue << "\t" << ENFP << "\t" << significant << std::endl;

    // Reset the set size before the next loop iteration
    setSize = 0;
  }

  // Free all the arrays made on the GPU
  free(randomValues);

  return 0;
}


//-----------------------------------------------------------------------------------
// HELPER PROCESSING FUNCTIONS
//-----------------------------------------------------------------------------------

/*
 * Populate an array (qIndex), the same length as the total number of genes involved,
 * which contains zeros if the gene is not part of the signature, 1 of it is upregulated, -1 if it is downregulated
 */
int queryToIndex(int *qIndex, const std::vector<std::string> &sigGeneNameList, const std::vector<int> &sigRegValue, const std::vector<std::string> &refGeneNameList) {
  int nMatches = 0;
  int nSigNames = sigGeneNameList.size();
  for (int r = 0; r < refGeneNameList.size(); r++) {
    // Loop through the gene name list
    // Extract the gene name to look for
    const std::string &refGeneName = refGeneNameList[r];
    // Find out if the gene name is anywhere within the reference gene name list
    for (int g = 0; g < sigGeneNameList.size(); g++) {
      // If we find a match, store within qIndex the up/down regulation value for the gene
      // at the location or the match within the gene name list
      if (refGeneName.compare(sigGeneNameList[g]) == 0) {
        qIndex[r] = sigRegValue[g];
        // Increment the match counter
        nMatches++;
        // If we've now found all the genes we need to, we can return
        if (nMatches == nSigNames)
          return 0;
        break;
      }
    }
  }
  // If we reach here, it implies we didn't find all the genes we want - this is bad
  std::cout << "nSigNames: " << nSigNames << ", nMatches: " << nMatches << std::endl;
  return -1;
}


/*
 * Get the total number of drugs - which here is dependent upon the value of compoundChoice
 */
inline int getNDrugs(const int compoundChoice) {
  switch (compoundChoice) {
    case 1:
      return 1309;
    case 2:
      return 1409;
    case 3:
      return 3738;
    case 4:
      return 6100;
  }
  return -1; // Unknown
}


/*
 * Compute the p-value using connection scores from random gene lists
 */
double computePValue(
  queue &q,
  const int nRandomGenerations, 
  const int threadsPerBlock,
  const double averageSetScore,
  const int setSize,
  const int signatureByRNGs,
  const double UCmax,
  buffer<float, 1> &device_randomIndexArray,
  buffer<int, 1> &device_refRegNum,
  buffer<float, 1> &device_arraysAdded) 
{
  // Figure out how many blocks are needed, given the threads per block
  const int blocksPerGrid = (int)ceil((float)nRandomGenerations / (float)threadsPerBlock);
   
  range<1> gws (blocksPerGrid * threadsPerBlock);
  range<1> lws (threadsPerBlock);

  // Compute scores of random gene signatures
  q.submit([&] (handler &cgh) {
    auto ria = device_randomIndexArray.get_access<sycl_read>(cgh);
    auto rrn = device_refRegNum.get_access<sycl_read>(cgh);
    auto arr = device_arraysAdded.get_access<sycl_write>(cgh);
    cgh.parallel_for<class compute_score>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      computeRandomConnectionScores(item,
        ria.get_pointer(), rrn.get_pointer(), arr.get_pointer(), 
        signatureByRNGs, UCmax, setSize, nRandomGenerations);
    });
  });

  // Start working out the P-value on the GPU by calculating the sum of the instances in which the random gene signature score was >= the query signature score
  buffer<int, 1> device_aboveThresholdAccumulator (blocksPerGrid);

  q.submit([&] (handler &cgh) {
    auto aar = device_arraysAdded.get_access<sycl_read>(cgh);
    auto ata = device_aboveThresholdAccumulator.get_access<sycl_write>(cgh);
    accessor<int, 1, sycl_read_write, access::target::local> sm (threadsPerBlock, cgh);
    cgh.parallel_for<class count>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      countAboveThresholdHelper(item, sm.get_pointer(),
        aar.get_pointer(), averageSetScore, ata.get_pointer(), nRandomGenerations);
    });
  });

  // Copy the array output from the GPU to the host, sum it and free the memory
  int *aboveThresholdAccumulator= new int[blocksPerGrid];

  q.submit([&] (handler &cgh) {
    auto acc = device_aboveThresholdAccumulator.get_access<sycl_read>(cgh);
    cgh.copy(acc, aboveThresholdAccumulator);
  }).wait();

  int aboveThresholdSum = 0;
  for (int ii = 0; ii < blocksPerGrid; ii++)
    aboveThresholdSum += aboveThresholdAccumulator[ii];

  // Finally calculate the P-value based on the GPU results and how many random numbers were generated
  delete [] aboveThresholdAccumulator;
  return computePValueHelper(aboveThresholdSum, nRandomGenerations);
}


/*
 * Convert the number of above-threshold connection scores into a p-value
 */
double computePValueHelper(const double nAboveThreshold, const int nRandomGenerations) {

  double pValueR = 0.0;
  double pValueL = 0.0;
  double pValue = 0.0;

  if (nAboveThreshold < 1) {
    pValueL = (0.5 / nRandomGenerations);
    pValueR = ((nRandomGenerations - 0.5) / nRandomGenerations);
  }
  else if (nAboveThreshold > nRandomGenerations-1) {
    pValueR = (0.5 / nRandomGenerations);
    pValueL = (nRandomGenerations - 0.5) / nRandomGenerations;
  }
  else {
    pValueL = nAboveThreshold / nRandomGenerations;
    pValueR = (nRandomGenerations - nAboveThreshold) / nRandomGenerations;
  }

  if (pValueR < pValueL)
    pValue = pValueR * 2;

  if (pValueR > pValueL)
    pValue = pValueL * 2;

  return pValue;
}

/*
 * Compute cMax for an unordered list of genes
 */
inline double computeUCMax(const int sigNGenes, const int nGenesTotal) {
  return ((sigNGenes * nGenesTotal) - (sigNGenes * (sigNGenes + 1))/2 + sigNGenes);
}

/*
 * Compute the dot product of two vectors, using the GPU
 */
double computeDotProduct(
  queue &q,
  buffer<int, 1> &device_v1,
  buffer<int, 1> &device_v2,
  const int vLength, const int blockSize, const int nThreads)
{
  // Compute dot product of device_qIndex and device_refRegNum using the GPU
  buffer<int, 1> device_temp (blockSize);

  range<1> gws (blockSize *nThreads);
  range<1> lws (nThreads);

  q.submit([&] (handler &cgh) {
    auto t = device_temp.get_access<sycl_write>(cgh);
    auto v1 = device_v1.get_access<sycl_read>(cgh);
    auto v2 = device_v2.get_access<sycl_read>(cgh);
    accessor<int, 1, sycl_read_write, access::target::local> sm (nThreads, cgh);
    cgh.parallel_for<class dotproduct>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      computeDotProductHelper (item,
        sm.get_pointer(), t.get_pointer(), v1.get_pointer(), v2.get_pointer(), vLength);
    });
  });

  // Copy the initial results back to the CPU for final summation
  int *result= new int[blockSize * sizeof(int)];
  q.submit([&] (handler &cgh) {
    auto acc = device_temp.get_access<sycl_read>(cgh);
    cgh.copy(acc, result);
  }).wait();
  
  double dot = 0.0;
  for (int z = 0; z < blockSize; z++) {
    dot += result[z];
  }
  delete [] result;
  return dot;
}

