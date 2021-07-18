#ifndef __UTILS_H
#define __UTILS_H

#include "common.h"

// Use double precision for accuracy 
#define float_sw4 double

void curvilinear4sg_ci(
    queue &q,
    int ifirst, int ilast, 
    int jfirst, int jlast, 
    int kfirst, int klast,
    buffer<float_sw4,1> &d_u, 
    buffer<float_sw4,1> &d_mu,
    buffer<float_sw4,1> &d_lambda,
    buffer<float_sw4,1> &d_met,
    buffer<float_sw4,1> &d_jac,
    buffer<float_sw4,1> &d_lu, 
    int* onesided,
    buffer<float_sw4,1> &d_cof, 
    buffer<float_sw4,1> &d_sg_str, 
    //float_sw4* d_acof, 
    //float_sw4* d_bope,
    //float_sw4* d_ghcof, 
    //float_sw4* d_acof_no_gp,
    //float_sw4* d_ghcof_no_gp, 
    //float_sw4* d_strx,
    //float_sw4* d_stry, 
    int nk, char op);

#endif
