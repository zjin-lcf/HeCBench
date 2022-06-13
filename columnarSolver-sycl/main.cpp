/*
Copyright 1998–2018 Bernhard Esslinger and the CrypTool Team. Permission
is granted to copy, distribute and/or modify this document under the terms of the
GNU Free Documentation License, Version 1.3 or any later version published by the
Free Software Foundation (FSF). A copy of the license is included in the section
entitled "GNU Free Documentation License".
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "common.h"

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


#include "kernels.cpp"

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
  if (argc != 2) {
    printf("Usage: %s <path to file>\n", argv[0]);
    return 1;
  }
  const char* filename = argv[1];

  int encryptedMap[ENCRYPTEDLEN];

  for (int j=0; j<ENCRYPTEDLEN; ++j)
    encryptedMap[j] = ENCRYPTED_T[j] - 'a';

  float scores[totalBigrams];  
  extractBigrams(scores, filename);

  int* decrypted = new int[ENCRYPTEDLEN*THREADS];

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float, 1> d_scores (scores, totalBigrams);
  buffer<int, 1> d_encrypted(encryptedMap, ENCRYPTEDLEN);
  buffer<int, 1> d_decrypted(decrypted, ENCRYPTEDLEN * THREADS);
  buffer<unsigned int, 1> d_states (THREADS);

  range<1> gws(THREADS);
  range<1> lws(T);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (handler &cgh) {
    auto states = d_states.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class setup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      setupKernel(item, states.get_pointer());
    });
  });

  q.submit([&] (handler &cgh) {
    auto scores = d_scores.get_access<sycl_read>(cgh);
    auto encrypted = d_encrypted.get_access<sycl_read>(cgh);
    auto states = d_states.get_access<sycl_read>(cgh);
    auto decrypted = d_decrypted.get_access<sycl_discard_write>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> lscores (ALPHABET*ALPHABET, cgh);
    cgh.parallel_for<class decode>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      decodeKernel(item, 
                   scores.get_pointer(), 
                   encrypted.get_pointer(),
                   states.get_pointer(),
                   decrypted.get_pointer(),
                   lscores.get_pointer());
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

  } // sycl scope

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

  delete[] decrypted;
  delete[] scoreHistory;
  return 0;
}
