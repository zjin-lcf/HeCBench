/*
 * Regular expression implementation.
 * Supports only ( | ) * + ?.  No escapes.
 * Compiles to NFA and then simulates NFA
 * using Thompson's algorithm.
 *
 * See also http://swtch.com/~rsc/regexp/ and
 * Thompson, Ken.  Regular Expression Search Algorithm,
 * Communications of the ACM 11(6) (June 1968), pp. 419-422.
 * 
 * Copyright (c) 2007 Russ Cox.
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the
 * Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall
 * be included in all copies or substantial portions of the
 * Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
 * KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS
 * OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "common.h"
#include "pnfa.h"
#include "cycleTimer.h"
#include "pnfa.cpp"


int checkCmdLine(int argc, char **argv, char **fileName, char **regexFile, int *time) {
  int visualize, simplified, postfix;
  SimpleReBuilder builder;
  State *start;

  parseCmdLine(argc, argv, &visualize, &postfix, time, &simplified, fileName, regexFile);

  // argv index at which regex is present
  int regexIndex = 1 + visualize + postfix + *time + simplified;
  if (fileName != NULL)
    regexIndex += 2;

  if (argc <= regexIndex) {
    usage (argv[0]);
    exit(EXIT_SUCCESS);
  }

  char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
  strcpy(regexBuffer, argv[regexIndex]);
  simplifyRe(&regexBuffer, &builder);
  free(regexBuffer);

  char *post = re2post(builder.re);
  if(post == NULL){
    fprintf(stderr, "bad regexp %s\n", argv[regexIndex]);
    return 1;
  }

  if (simplified == 1) {
    char * clean_simplified = stringifyRegex(builder.re);
    printf("\nSimplified Regex: %s\n", clean_simplified);
    free(clean_simplified);
    exit(0);
  }

  /* destruct the simpleRe */
  _simpleReBuilder(&builder);

  if (postfix == 1) {
    char * clean_post = stringifyRegex(post);
    printf("\nPostfix buffer: %s\n", clean_post);
    free(clean_post);
    exit(0);
  }

  if (visualize == 1) { 
    start = post2nfa(post);
    visualize_nfa(start);
    exit(0);
  }

  return regexIndex;

}


int main(int argc, char **argv)
{
  int timerOn;
  char *fileName = NULL, *regexFile = NULL, **lines = NULL;
  int num_lines;

  SimpleReBuilder builder;
  double startTime, endReadFile, 
         endSetup, endKernel, endTime;

  int regexIndex = checkCmdLine(argc, argv, &fileName, &regexFile, &timerOn);

  // parallel matching
  if (fileName == NULL) {
    printf("Enter a file \n");
    exit(EXIT_SUCCESS);
  }

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  //===============================================================================
  // match just a single regex  on a device
  //===============================================================================
  startTime = CycleTimer::currentSeconds();  
  readFile(fileName, &lines, &num_lines);    
  endReadFile = CycleTimer::currentSeconds();

  char * regexBuffer = (char*)malloc(strlen(argv[regexIndex])+1);
  strcpy(regexBuffer, argv[regexIndex]);
  simplifyRe(&regexBuffer, &builder);
  free(regexBuffer);

  int postsize = (strlen(builder.re) + 1) * sizeof (char);

  buffer<char, 1> device_regex(builder.re, postsize);

  u32 * table = (u32 *) malloc(sizeof(u32) * strlen(*lines));
  table[0] = 0;
  num_lines = 0;

  int len = strlen(lines[0]);
  for (int i = 0; i < len; i++) {
    if ((lines[0])[i] == '\n') {
      table[++num_lines] = i+1;
      lines[0][i] = 0;    
    }
  }

  if((lines[0])[len-1] == '\n')/*if at the end file not '\n', then we not forgot last offset */
    --num_lines;

  buffer<u32, 1> device_line_table(table, len);
  buffer<char, 1> device_line(*lines, len+1);

  u32 host_regex_table[1]; /*offsets to regexes on host*/
  host_regex_table[0]=0;   /*in case of one regex offset must be 0*/

  buffer<u32, 1> device_regex_table(host_regex_table, 1);

  buffer<unsigned char, 1> device_result(num_lines);

  State pmatchstate = { Match };  /* matching state */
  buffer<State, 1> device_match_state(&pmatchstate, 1);

  endSetup = CycleTimer::currentSeconds();

  // measure kernel execution time
  q.submit([&] (handler &cgh) {
    auto bigLine = device_line.get_access<sycl_read_write>(cgh);
    auto tableOfLineStarts = device_line_table.get_access<sycl_read>(cgh);
    auto regexLines = device_regex.get_access<sycl_read_write>(cgh);
    auto regexTable = device_regex_table.get_access<sycl_read>(cgh);
    auto devResult = device_result.get_access<sycl_discard_write>(cgh);
    auto pmatchstate = device_match_state.get_access<sycl_read_write>(cgh);

    accessor<char, 1, sycl_read_write, access::target::local> buf(BUFFER_SIZE, cgh);
    accessor<int, 1, sycl_read_write, access::target::local> pnstate(1, cgh);
    accessor<State, 1, sycl_read_write, access::target::local> s(100, cgh);
    accessor<State*, 1, sycl_read_write, access::target::local> st(1, cgh);

    cgh.parallel_for<class regExpMatch>(nd_range<1>(range<1>(512*160), range<1>(160)), [=] (nd_item<1> item) {
      if (item.get_local_id(0) == 0) {
        pre2post(regexLines.get_pointer() + regexTable[0], buf.get_pointer());

        pnstate[0] = 0;
        st[0] = ppost2nfa(buf.get_pointer(), s.get_pointer(), 
                          pnstate.get_pointer(), pmatchstate.get_pointer());
      }

      item.barrier(access::fence_space::local_space);

      List d1;
      List d2;  

      int i;
      for (i = item.get_global_id(0); i < num_lines;
           i += item.get_local_range(0) * item.get_group_range(0)) { 

        char * lineSegment = bigLine.get_pointer() + tableOfLineStarts[i];
        if (panypmatch(st[0], lineSegment, &d1, &d2)) 
          devResult[i] = 1;
        else
          devResult[i] = 0;
      }
    });  
  });  

  q.wait();

  endKernel = CycleTimer::currentSeconds();

  unsigned char *host_result = (unsigned char *) malloc (num_lines * sizeof(unsigned char));
  q.submit([&] (handler &cgh) {
    auto acc = device_result.get_access<sycl_read>(cgh);
    cgh.copy(acc, host_result);
  }).wait();

  // print the "grep" results for verification
  if (!timerOn) {  
    for (int i = 0; i < num_lines; i++) {
      if(host_result[i] == 1) 
        PRINT(timerOn, "%s\n", lines[0] + table[i]);
    }
  }

  free(table);
  free(host_result);
  free(*lines);
  free(lines);

  endTime = CycleTimer::currentSeconds();

  if (timerOn) {
    printf("\nReadFile time %.4f \n", (endReadFile - startTime));
    printf("\nDevice setup time %.4f \n", (endSetup - endReadFile));
    printf("\nKernel execution Time %.4f \n", (endKernel - endSetup));
    printf("\nTotal time %.4f \n\n", (endTime - startTime));
  }

  return EXIT_SUCCESS;
}


