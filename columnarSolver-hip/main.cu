#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime.h>

#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)

#define ENCRYPTED_T "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

#define KEY_LENGTH 30
#define SECTION_CONSTANT ENCRYPTEDLEN/KEY_LENGTH

#define HEUR_THRESHOLD_OP1 50
#define HEUR_THRESHOLD_OP2 70

#define OP1_HOP 4
#define OP2_HOP 2


#include "kernels.cu"

void extractBigrams(float *scores, const char* filename) {
  FILE* bigramsFile = fopen(filename, "r");
  while(1){
    char tempBigram[2];
    float tempBigramScore = 0.0;
    if (fscanf(bigramsFile, "%s %f", tempBigram, &tempBigramScore) < 2)
    { break; } 
    scores[(tempBigram[0]-'a')*ALPHABET + tempBigram[1]-'a'] = tempBigramScore; 
  }
  fclose(bigramsFile);
}

bool verify(int* encrMap) {
  bool pass = true;
  const char *expect = DECRYPTED_T;
  for (int j=0; j<ENCRYPTEDLEN; ++j) {
    if (encrMap[j] + 'a' != expect[j]) {
       pass = false; break;
    }
  }
  return pass;
}

float candidateScore(int* decrMsg, float* scores) {
  float total = 0.0;
  for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
    total += scores[ALPHABET*decrMsg[j] + decrMsg[j+1]];  
  return total;
}


int main(int argc, char* argv[]) {

  const char* filename = argv[1];

  int encryptedMap[ENCRYPTEDLEN];

  for (int j=0; j<ENCRYPTEDLEN; ++j)
    encryptedMap[j] = ENCRYPTED_T[j] - 'a';

  float scores[totalBigrams];  
  extractBigrams(scores, filename);

  float *d_scores;
  int *d_encrypted, *d_decrypted;
  int* decrypted = new int[ENCRYPTEDLEN*THREADS];

  hipMalloc((void **)&d_scores, sizeof(float)*totalBigrams);
  hipMalloc((void **)&d_encrypted, sizeof(int)*ENCRYPTEDLEN);
  hipMalloc((void **)&d_decrypted, sizeof(int)*ENCRYPTEDLEN*THREADS);

  hipMemcpy(d_scores, scores, sizeof(float)*totalBigrams, hipMemcpyHostToDevice);
  hipMemcpy(d_encrypted, encryptedMap, sizeof(int)*ENCRYPTEDLEN, hipMemcpyHostToDevice);

  unsigned int* devStates;
  hipMalloc(&devStates, THREADS*sizeof(unsigned int));

  hipLaunchKernelGGL(setupKernel, dim3(B), dim3(T), 0, 0, devStates);

  hipLaunchKernelGGL(decode, dim3(B), dim3(T), 0, 0, d_scores, d_encrypted, devStates, d_decrypted);

  hipMemcpy(decrypted, d_decrypted, sizeof(int)*ENCRYPTEDLEN*THREADS, hipMemcpyDeviceToHost);

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  //  Calculating the best score ..
  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*j], scores);
    scoreHistory[j] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }    
  }  

  //printf("Best candidate score: %f\n", bestScore);
  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  hipFree(d_scores);
  hipFree(d_encrypted);
  hipFree(d_decrypted);
  hipFree(devStates);
  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
}
