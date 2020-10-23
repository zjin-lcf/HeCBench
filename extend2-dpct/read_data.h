#ifndef __READ_DATA
#define __READ_DATA

#include <stdint.h>

struct extend2_dat {
	/* input */
	int qlen;
	unsigned char *query;
	int tlen;
	unsigned char *target;
	int m;
	char mat[25];
	int o_del;
	int e_del;
	int o_ins;
	int e_ins;
	int w;
	int end_bonus;
	int zdrop;
	int h0;
	/* output */
	int qle;
	int tle;
	int gtle;
	int gscore;
	int max_off;
	/* return */
	int score;
	/* time-to-solution in cycle */
	uint64_t tsc;
	double sec;
};

extern int read_data(const char *fn, struct extend2_dat *d);

#endif
