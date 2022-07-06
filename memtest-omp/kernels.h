#define TYPE unsigned long
#define MAX_ERR_RECORD_COUNT 10
#define BLOCKSIZE (1024*1024)

//each thread is responsible for 1 BLOCKSIZE each time
void kernel0_write(char*__restrict ptr, unsigned long size)
{
  unsigned long n = size/BLOCKSIZE;

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    unsigned long * start_p = (unsigned long*)(ptr + i*BLOCKSIZE);
    unsigned long * end_p = (unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    unsigned long * p = start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;

    *p = pattern;
    pattern = (pattern << 1);
    while(p < end_p){
      p = (unsigned long*) (((unsigned long)start_p)|mask);

      if(p == start_p) {
        mask = (mask << 1);
        if (mask == 0) break;
        continue;
      }

      if (p >= end_p) break;

      *p = pattern;
      pattern = pattern <<1;
      mask = (mask << 1);
      if (mask == 0) break;
    }
  }
}

void kernel0_read(
    const char*__restrict ptr, unsigned long size,
    unsigned int*__restrict err_count,
    unsigned long*__restrict err_addr,
    unsigned long*__restrict err_expect,
    unsigned long*__restrict err_current,
    unsigned long*__restrict err_second_read)
{
  unsigned long n = size/BLOCKSIZE;
  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    unsigned long * start_p = (unsigned long*)(ptr + i*BLOCKSIZE);
    unsigned long * end_p = (unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    unsigned long * p = start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;

    if (*p != pattern) {
      // RECORD_ERR(err_count, p, pattern, *p);
      unsigned int idx;
      #pragma omp atomic capture
      idx = err_count[0]++;
      idx = idx % MAX_ERR_RECORD_COUNT;
      err_addr[idx] = (unsigned long)p;
      err_expect[idx] = (unsigned long)pattern;
      err_current[idx] = (unsigned long)(*p);
      err_second_read[idx] = (unsigned long)(*p);
    }

    pattern = (pattern << 1);
    while (p< end_p) {
      p = ( unsigned long*)( ((unsigned long)start_p)|mask);

      if(p == start_p) {
        mask = (mask << 1);
        if (mask == 0) break;
        continue;
      }

      if (p >= end_p) break;

      if (*p != pattern) {
        unsigned int idx;
        #pragma omp atomic capture
        idx = err_count[0]++;
        idx = idx % MAX_ERR_RECORD_COUNT;
        err_addr[idx] = (unsigned long)p;
        err_expect[idx] = (unsigned long)pattern;
        err_current[idx] = (unsigned long)(*p);
        err_second_read[idx] = (unsigned long)(*p);
      }

      pattern = pattern <<1;
      mask = (mask << 1);
      if (mask == 0) break;
    }
  }
}


void kernel1_write(char* ptr, unsigned long size)
{
  unsigned long* buf = ( unsigned long*)ptr;
  unsigned long n = size/sizeof(unsigned long);

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++)
    buf[i] = (unsigned long)(buf+i);
}

void kernel1_read(
    const char*__restrict ptr, unsigned long size,
    unsigned int*__restrict err_count,
    unsigned long*__restrict err_addr,
    unsigned long*__restrict err_expect,
    unsigned long*__restrict err_current,
    unsigned long*__restrict err_second_read)
{
  unsigned long* buf = ( unsigned long*)ptr;
  unsigned long n = size/sizeof(unsigned long);

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    if(buf[i] != (unsigned long)(buf+i)) {
      // RECORD_ERR(err_count, &buf[i], (buf+i), buf[i]);
      unsigned int idx;
      #pragma omp atomic capture
      idx = err_count[0]++;
      idx = idx % MAX_ERR_RECORD_COUNT;
      err_addr[idx] = (unsigned long)(buf+i);
      err_expect[idx] = (unsigned long)(buf+i);
      err_current[idx] = (unsigned long)buf[i];
      err_second_read[idx] = (unsigned long)buf[i];
    }
  }
}

void kernel_write(char* ptr, unsigned long size, TYPE p1)
{
  TYPE* buf = (TYPE*)ptr;
  unsigned long n = size/sizeof(TYPE);

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++)
    buf[i] = p1;
}


void kernel_read_write(
    char*__restrict ptr, unsigned long size, TYPE p1, TYPE p2,
    unsigned int*__restrict err_count,
    unsigned long*__restrict err_addr,
    unsigned long*__restrict err_expect,
    unsigned long*__restrict err_current,
    unsigned long*__restrict err_second_read)
{
  TYPE* buf = (TYPE*) ptr;
  unsigned long n =  size/sizeof(TYPE);
  TYPE localp;

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {

    localp = buf[i];

    if (localp != p1) {
      // RECORD_ERR(err_count, &buf[i], p1, localp);
      unsigned int idx;
      #pragma omp atomic capture
      idx = err_count[0]++;
      idx = idx % MAX_ERR_RECORD_COUNT;
      err_addr[idx] = (unsigned long)(buf+i);
      err_expect[idx] = (unsigned long)p1;
      err_current[idx] = (unsigned long)localp;
      err_second_read[idx] = (unsigned long)buf[i];
    }

    buf[i] = p2;
  }
}

void kernel_read(
    const char*__restrict ptr, unsigned long size, TYPE p1,
    unsigned int*__restrict err_count,
    unsigned long*__restrict err_addr,
    unsigned long*__restrict err_expect,
    unsigned long*__restrict err_current,
    unsigned long*__restrict err_second_read)
{
  TYPE* buf = (TYPE*) ptr;
  unsigned long n =  size/sizeof(TYPE);
  TYPE localp;

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    localp = buf[i];
    if (localp != p1) {
      // RECORD_ERR(err_count, &buf[i], p1, localp);
      unsigned int idx;
      #pragma omp atomic capture
      idx = err_count[0]++;
      idx = idx % MAX_ERR_RECORD_COUNT;
      err_addr[idx] = (unsigned long)(buf+i);
      err_expect[idx] = (unsigned long)p1;
      err_current[idx] = (unsigned long)localp;
      err_second_read[idx] = (unsigned long)buf[i];
    }
  }
}

void kernel5_init(char* ptr, unsigned long size)
{
  unsigned int * buf = (unsigned int*)ptr;
  unsigned long n = size/64;

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    unsigned int p1 = 1;
    p1 = p1 << (i % 32);
    unsigned int p2 = ~p1;
    buf[i*16] = p1;
    buf[i*16 + 1] = p1;
    buf[i*16 + 2] = p2;
    buf[i*16 + 3] = p2;
    buf[i*16 + 4] = p1;
    buf[i*16 + 5] = p1;
    buf[i*16 + 6] = p2;
    buf[i*16 + 7] = p2;
    buf[i*16 + 8] = p1;
    buf[i*16 + 9] = p1;
    buf[i*16 + 10] = p2;
    buf[i*16 + 11] = p2;
    buf[i*16 + 12] = p1;
    buf[i*16 + 13] = p1;
    buf[i*16 + 14] = p2;
    buf[i*16 + 15] = p2;
  }
}

void kernel5_move(char* ptr, unsigned long size)
{
  unsigned long n = size/BLOCKSIZE;
  unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    unsigned int* mybuf = (unsigned int*)(ptr + i*BLOCKSIZE);
    unsigned int* mybuf_mid = (unsigned int*)(ptr + i*BLOCKSIZE + BLOCKSIZE/2);
    int j;
    for (j = 0; j < half_count; j++)
      mybuf_mid[j] = mybuf[j];

    for (j = 0;j < half_count -8; j++)
      mybuf[j+8] = mybuf_mid[j];

    for (j = 0; j < 8; j++)
      mybuf[j] = mybuf_mid[half_count - 8+j];
  }
}


void kernel5_check(
    const char*__restrict ptr, unsigned long size,
    unsigned int*__restrict err_count,
    unsigned long*__restrict err_addr,
    unsigned long*__restrict err_expect,
    unsigned long*__restrict err_current,
    unsigned long*__restrict err_second_read)
{
  unsigned int * buf = (unsigned int*)ptr;
  unsigned long n = size/(2*sizeof(unsigned int));

  #pragma omp target teams distribute parallel for num_teams(1024) thread_limit(64)
  for (int i = 0; i < n; i++) {
    if (buf[2*i] != buf[2*i+1]) {
      // RECORD_ERR(err_count, &buf[2*i], buf[2*i+1], buf[2*i]);
      unsigned int idx;
      #pragma omp atomic capture
      idx = err_count[0]++;
      idx = idx % MAX_ERR_RECORD_COUNT;
      err_addr[idx] = (unsigned long)(buf+2*i);
      err_expect[idx] = (unsigned long)buf[2*i+1];
      err_current[idx] = (unsigned long)buf[2*i];
      err_second_read[idx] = (unsigned long)buf[2*i];
    }
  }
}
