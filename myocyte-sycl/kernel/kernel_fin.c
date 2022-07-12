#include "../common.h"                      // (in directory)

void kernel_fin(
    FP *initvalu,
    int initvalu_offset_ecc,
    int initvalu_offset_Dyad,
    int initvalu_offset_SL,
    int initvalu_offset_Cyt,
    FP *parameter,
    FP *finavalu,
    FP JCaDyad,
    FP JCaSL,
    FP JCaCyt){

  //=====================================================================
  //  VARIABLES
  //=====================================================================

  // decoded input parameters
  FP BtotDyad;                                    //
  FP CaMKIItotDyad;                                  //

  // compute variables
  FP Vmyo;                                      // [L]
  FP Vdyad;                                      // [L]
  FP VSL;                                        // [L]
  // FP kDyadSL;                                      // [L/msec]
  FP kSLmyo;                                      // [L/msec]
  FP k0Boff;                                      // [s^-1] 
  FP k0Bon;                                      // [uM^-1 s^-1] kon = koff/Kd
  FP k2Boff;                                      // [s^-1] 
  FP k2Bon;                                      // [uM^-1 s^-1]
  // FP k4Boff;                                      // [s^-1]
  FP k4Bon;                                      // [uM^-1 s^-1]
  FP CaMtotDyad;
  FP Bdyad;                                      // [uM dyad]
  FP J_cam_dyadSL;                                  // [uM/msec dyad]
  FP J_ca2cam_dyadSL;                                  // [uM/msec dyad]
  FP J_ca4cam_dyadSL;                                  // [uM/msec dyad]
  FP J_cam_SLmyo;                                    // [umol/msec]
  FP J_ca2cam_SLmyo;                                  // [umol/msec]
  FP J_ca4cam_SLmyo;                                  // [umol/msec]

  //=====================================================================
  //  COMPUTATION
  //=====================================================================

  // decoded input parameters
  BtotDyad      = parameter[2];                            //
  CaMKIItotDyad = parameter[3];                            //

  // set variables
  Vmyo = 2.1454e-11;                                  // [L]
  Vdyad = 1.7790e-14;                                  // [L]
  VSL = 6.6013e-13;                                  // [L]
  // kDyadSL = 3.6363e-16;                                // [L/msec]
  kSLmyo = 8.587e-15;                                  // [L/msec]
  k0Boff = 0.0014;                                  // [s^-1] 
  k0Bon = k0Boff/0.2;                                  // [uM^-1 s^-1] kon = koff/Kd
  k2Boff = k0Boff/100;                                // [s^-1] 
  k2Bon = k0Bon;                                    // [uM^-1 s^-1]
  // k4Boff = k2Boff;                                  // [s^-1]
  k4Bon = k0Bon;                                    // [uM^-1 s^-1]

  // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
  finavalu[initvalu_offset_ecc+35] = finavalu[initvalu_offset_ecc+35] + 1e-3*JCaDyad;
  finavalu[initvalu_offset_ecc+36] = finavalu[initvalu_offset_ecc+36] + 1e-3*JCaSL;
  finavalu[initvalu_offset_ecc+37] = finavalu[initvalu_offset_ecc+37] + 1e-3*JCaCyt; 

  // incorporate CaM diffusion between compartments
  CaMtotDyad = initvalu[initvalu_offset_Dyad+0]
    + initvalu[initvalu_offset_Dyad+1]
    + initvalu[initvalu_offset_Dyad+2]
    + initvalu[initvalu_offset_Dyad+3]
    + initvalu[initvalu_offset_Dyad+4]
    + initvalu[initvalu_offset_Dyad+5]
    + CaMKIItotDyad * (  initvalu[initvalu_offset_Dyad+6]
        + initvalu[initvalu_offset_Dyad+7]
        + initvalu[initvalu_offset_Dyad+8]
        + initvalu[initvalu_offset_Dyad+9])
    + initvalu[initvalu_offset_Dyad+12]
    + initvalu[initvalu_offset_Dyad+13]
    + initvalu[initvalu_offset_Dyad+14];
  Bdyad = BtotDyad - CaMtotDyad;                                        // [uM dyad]
  J_cam_dyadSL = 1e-3 * (  k0Boff*initvalu[initvalu_offset_Dyad+0] - k0Bon*Bdyad*initvalu[initvalu_offset_SL+0]);      // [uM/msec dyad]
  J_ca2cam_dyadSL = 1e-3 * (  k2Boff*initvalu[initvalu_offset_Dyad+1] - k2Bon*Bdyad*initvalu[initvalu_offset_SL+1]);    // [uM/msec dyad]
  J_ca4cam_dyadSL = 1e-3 * (  k2Boff*initvalu[initvalu_offset_Dyad+2] - k4Bon*Bdyad*initvalu[initvalu_offset_SL+2]);    // [uM/msec dyad]

  J_cam_SLmyo = kSLmyo * (  initvalu[initvalu_offset_SL+0] - initvalu[initvalu_offset_Cyt+0]);                // [umol/msec]
  J_ca2cam_SLmyo = kSLmyo * (  initvalu[initvalu_offset_SL+1] - initvalu[initvalu_offset_Cyt+1]);              // [umol/msec]
  J_ca4cam_SLmyo = kSLmyo * (  initvalu[initvalu_offset_SL+2] - initvalu[initvalu_offset_Cyt+2]);              // [umol/msec]

  // ADJUST CAM Dyad 
  finavalu[initvalu_offset_Dyad+0] = finavalu[initvalu_offset_Dyad+0] - J_cam_dyadSL;
  finavalu[initvalu_offset_Dyad+1] = finavalu[initvalu_offset_Dyad+1] - J_ca2cam_dyadSL;
  finavalu[initvalu_offset_Dyad+2] = finavalu[initvalu_offset_Dyad+2] - J_ca4cam_dyadSL;

  // ADJUST CAM Sl
  finavalu[initvalu_offset_SL+0] = finavalu[initvalu_offset_SL+0] + J_cam_dyadSL*Vdyad/VSL - J_cam_SLmyo/VSL;
  finavalu[initvalu_offset_SL+1] = finavalu[initvalu_offset_SL+1] + J_ca2cam_dyadSL*Vdyad/VSL - J_ca2cam_SLmyo/VSL;
  finavalu[initvalu_offset_SL+2] = finavalu[initvalu_offset_SL+2] + J_ca4cam_dyadSL*Vdyad/VSL - J_ca4cam_SLmyo/VSL;

  // ADJUST CAM Cyt 
  finavalu[initvalu_offset_Cyt+0] = finavalu[initvalu_offset_Cyt+0] + J_cam_SLmyo/Vmyo;
  finavalu[initvalu_offset_Cyt+1] = finavalu[initvalu_offset_Cyt+1] + J_ca2cam_SLmyo/Vmyo;
  finavalu[initvalu_offset_Cyt+2] = finavalu[initvalu_offset_Cyt+2] + J_ca4cam_SLmyo/Vmyo;

}
