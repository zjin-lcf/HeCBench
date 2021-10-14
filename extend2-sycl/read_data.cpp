/*
   written by Kaz Yoshii kazutomo.yoshii@gmail
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "read_data.h"

/* return non-zero on success */
static int read_check(int fd, void *buf, size_t sz)
{
  if(read(fd, buf, sz) != (ssize_t)sz) return 0;
  return 1;
}

/* return non-zero on success */
int read_data(const char *fn, struct extend2_dat *d)
{
  if (!d) return 0;
  int fd = open(fn, O_RDONLY);
  if (fd < 0) {
    perror("Cannot open output file\n");
    return 0;
  }

  /* input */
  assert( read_check(fd, (void*)&d->qlen, sizeof(int)) );
  d->query = (uint8_t*)malloc(d->qlen);
  assert( read_check(fd, d->query, d->qlen) );
  assert( read_check(fd, (void*)&d->tlen, sizeof(int)) );
  d->target = (uint8_t*)malloc(d->tlen);
  assert( read_check(fd, d->target, d->tlen) );
  assert( read_check(fd, (void*)&d->m, sizeof(int)) );
  assert( read_check(fd, (void*)d->mat, 25) );
  assert( read_check(fd, (void*)&d->o_del, sizeof(int)) );
  assert( read_check(fd, (void*)&d->e_del, sizeof(int)) );
  assert( read_check(fd, (void*)&d->o_ins, sizeof(int)) );
  assert( read_check(fd, (void*)&d->e_ins, sizeof(int)) );
  assert( read_check(fd, (void*)&d->w, sizeof(int)) );
  assert( read_check(fd, (void*)&d->end_bonus, sizeof(int)) );
  assert( read_check(fd, (void*)&d->zdrop, sizeof(int)) );
  assert( read_check(fd, (void*)&d->h0, sizeof(int)) );

  /* results */
  assert( read_check(fd, (void*)&d->qle, sizeof(int)) );
  assert( read_check(fd, (void*)&d->tle, sizeof(int)) );
  assert( read_check(fd, (void*)&d->gtle, sizeof(int)) );
  assert( read_check(fd, (void*)&d->gscore, sizeof(int)) );
  assert( read_check(fd, (void*)&d->max_off, sizeof(int)) );

  /* return */
  assert( read_check(fd, (void*)&d->score, sizeof(int)) );

  /* time-to-solution in cycle */
  assert( read_check(fd, (void*)&d->tsc, sizeof(uint64_t)) );
  assert( read_check(fd, (void*)&d->sec, sizeof(double)) );

  close(fd);
  return 1;
}

