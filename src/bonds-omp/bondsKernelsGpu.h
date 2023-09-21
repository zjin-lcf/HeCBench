//bondsKernelsGpu.cuh
//Scott Grauer-Gray
//Header for bonds kernels to run on the GPU

#ifndef BONDS_KERNELS_GPU
#define BONDS_KERNELS_GPU

#include <stdbool.h>
#include "bondsStructs.h"

int monthLengthKernelGpu(int month, bool leapYear);

int monthOffsetKernelGpu(int m, bool leapYear);

int yearOffsetKernelGpu(int y);

bool isLeapKernelGpu(int y);

bondsDateStruct intializeDateKernelGpu(int d, int m, int y);

dataType yearFractionGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);

int dayCountGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter);

dataType couponNotionalGpu();

dataType bondNotionalGpu();

dataType fixedRateCouponNominalGpu();

bool eventHasOccurredGpu(bondsDateStruct currDate, bondsDateStruct eventDate);

bool cashFlowHasOccurredGpu(bondsDateStruct refDate, bondsDateStruct eventDate);

bondsDateStruct advanceDateGpu(bondsDateStruct date, int numMonthsAdvance);

dataType getAccruedAmountGpu(bondsDateStruct* maturityDate, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs);

dataType fixedRateCouponAccruedAmountGpu(cashFlowsStruct cashFlows, int numLeg, 
bondsDateStruct d, bondsDateStruct* maturityDate, int bondNum);

dataType cashFlowsNpvGpu(cashFlowsStruct cashFlows,
    bondsYieldTermStruct discountCurve,
    bool includeSettlementDateFlows,
    bondsDateStruct settlementDate,
    bondsDateStruct npvDate,
    int numLegs);

dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct, bondsDateStruct t);

dataType flatForwardDiscountImplGpu(intRateStruct intRate, dataType t);

dataType interestRateDiscountFactorGpu(intRateStruct intRate, dataType t);

dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t);

dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg);

dataType interestRateCompoundFactorGpu(intRateStruct intRate, bondsDateStruct d1,
    bondsDateStruct d2, int dayCounter);

dataType interestRateImpliedRateGpu(dataType compound,                                        
    int comp,
    dataType freq,
    dataType t);

int cashFlowsNextCashFlowNumGpu(cashFlowsStruct cashFlows,
    bondsDateStruct settlementDate,
    int numLegs);

dataType getBondYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondStruct *bond,
    bondsDateStruct *maturityDate,
 int bondNum, cashFlowsStruct cashFlows, int numLegs);


dataType getBondFunctionsYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondsDateStruct* maturityDate,
int bondNum, cashFlowsStruct cashFlows, int numLegs);

dataType solverSolveGpu(solverStruct solver,
    irrFinderStruct f,
    dataType accuracy,
    dataType guess,
    dataType step,
    cashFlowsStruct cashFlows,
    int numLegs);

dataType cashFlowsAccruedAmountGpu(cashFlowsStruct cashFlows,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    int numLegs, bondsDateStruct* maturityDate, int bondNum);

dataType cashFlowsNpvYieldGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includeSettlementDateFlows,
    bondsDateStruct settlementDate,
    bondsDateStruct npvDate,
    int numLegs);

dataType fOpGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);

dataType fDerivativeGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);

bool closeGpu(dataType x, dataType y);

bool closeGpuThreeArgs(dataType x, dataType y, int n);

dataType solveImplGpu(solverStruct solver, irrFinderStruct f,
    dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs);

dataType modifiedDurationGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includeSettlementDateFlows,
    bondsDateStruct settlementDate,
    bondsDateStruct npvDate,
    int numLegs);

dataType getCashFlowsYieldGpu(cashFlowsStruct cashFlows,
    dataType npv,
    int dayCounter,
    int compounding,
    dataType frequency,
    bool includeSettlementDateFlows,
    bondsDateStruct settlementDate,
    bondsDateStruct npvDate,
    int numLegs,
    dataType accuracy/* = 1.0e-10*/,
    int maxIterations/* = 100*/,
    dataType guess/* = 0.05f*/);

#endif //BONDS_KERNELS_GPU

