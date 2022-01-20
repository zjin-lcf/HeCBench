#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include "utils.h"
#include "kernel.h"

int main() {
  float *th, *pii, *q;
  float *qc, *qi, *qr, *qs;
  float *den, *p, *delz;
  float *rain,*rainncv;
  float *sr;
  float *snow, *snowncv;

  float delt = 10.f;
  int ims = 0, ime = 59, jms = 0, jme = 45, kms = 0, kme = 2;
  int ips = 0, ipe = 59, jps = 0, jpe = 45, kps = 0, kpe = 2;
  int d3 = (ime-ims+1) * (jme-jms+1) * (kme-kms+1) ;
  int d2 = (ime-ims+1) * (jme-jms+1) ;

  int dips = 0 ; int dipe = (ipe-ips+1) ;
  int djps = 0 ; int djpe = (jpe-jps+1) ;
  int dkps = 0 ; int dkpe = (kpe-kps+1) ;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue Q(dev_sel);

  srand(123); // init. data randomly in TODEV macros
  TODEV3(th) ;
  TODEV3(pii) ;
  TODEV3(q) ;
  TODEV3(qc) ;
  TODEV3(qi) ;
  TODEV3(qr) ;
  TODEV3(qs) ;
  TODEV3(den) ;
  TODEV3(p) ;
  TODEV3(delz) ;
  TODEV2(rain) ;
  TODEV2(rainncv) ;
  TODEV2(sr) ;
  TODEV2(snow) ;
  TODEV2(snowncv) ;

  int remx = (ipe-ips+1) % XXX != 0 ? 1 : 0 ;
  int remy = (jpe-jps+1) % YYY != 0 ? 1 : 0 ;

  range<2> lws ( YYY, XXX ) ;
  range<2> gws ( YYY * ((jpe-jps+1) / YYY + remy),
                 XXX * ((ipe-ips+1) / XXX + remx) );

  Q.submit([&] (handler &cgh) {
    auto th = th_d.get_access<sycl_read_write>(cgh);
    auto pii = pii_d.get_access<sycl_read>(cgh);
    auto q = q_d.get_access<sycl_read_write>(cgh);
    auto qc = qc_d.get_access<sycl_read_write>(cgh);
    auto qi = qi_d.get_access<sycl_read_write>(cgh);
    auto qr = qr_d.get_access<sycl_read_write>(cgh);
    auto qs = qs_d.get_access<sycl_read_write>(cgh);
    auto den = den_d.get_access<sycl_read>(cgh);
    auto p = p_d.get_access<sycl_read>(cgh);
    auto delz = delz_d.get_access<sycl_read>(cgh);
    auto rain = rain_d.get_access<sycl_read_write>(cgh);
    auto rainncv = rainncv_d.get_access<sycl_read_write>(cgh);
    auto sr = sr_d.get_access<sycl_read_write>(cgh);
    auto snow = snow_d.get_access<sycl_read_write>(cgh);
    auto snowncv = snowncv_d.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class wp>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      wsm (item, th.get_pointer(),
           pii.get_pointer(),
           q.get_pointer(),
           qc.get_pointer(),
           qi.get_pointer(),
           qr.get_pointer(),
           qs.get_pointer(),
           den.get_pointer(),
           p.get_pointer(),
           delz.get_pointer(),
           rain.get_pointer(),
           rainncv.get_pointer(),
           sr.get_pointer(),
           snow.get_pointer(),
           snowncv.get_pointer(),
           delt,
           dips+1 , (ipe-ips+1) , // ids, ide
           djps+1 , (jpe-jps+1) , // jds, jde 
           dkps+1 , (kpe-kps+1),  // kds, kde
           dips+1 , dipe ,        // ims, ime
           djps+1 , djpe ,        // jms, jme
           dkps+1 , dkpe,         // kms, kme
           dips+1 , dipe ,        // ips, ipe
           djps+1 , djpe ,        // jps, jpe
           dkps+1 , dkpe) ;       // kps, kpe 
    });
  });

  Q.wait();

  FROMDEV2(rain) ;
  FROMDEV2(snow) ;

  // print rain and snow data
  float rain_sum = 0, snow_sum = 0;
  for (int i = 0; i < d2; i++) {
    rain_sum += rain[i];
    snow_sum += snow[i];
  }
  printf("Checksum: rain = %f snow = %f\n", rain_sum, snow_sum);

  FREE(th) ;
  FREE(pii) ;
  FREE(q) ;
  FREE(qc) ;
  FREE(qi) ;
  FREE(qr) ;
  FREE(qs) ;
  FREE(den) ;
  FREE(p) ;
  FREE(delz) ;
  FREE(rain) ;
  FREE(rainncv) ;
  FREE(sr) ;
  FREE(snow) ;
  FREE(snowncv) ;
  return(0) ;
}

