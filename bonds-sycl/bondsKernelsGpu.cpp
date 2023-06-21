//bondsKernelsGpu.cu
//Scott Grauer-Gray
//Bonds kernels to run on the GPU

#include "bondsKernelsGpu.h"

#define INLINE

INLINE
int monthLengthKernelGpu(int month, bool leapYear) 
{
  int MonthLength[12];
  MonthLength[0]=31;
  MonthLength[1]=28;
  MonthLength[2]=31;
  MonthLength[3]=30;
  MonthLength[4]=31;
  MonthLength[5]=30;
  MonthLength[6]=31;
  MonthLength[7]=31;
  MonthLength[8]=30;
  MonthLength[9]=31;
  MonthLength[10]=30;
  MonthLength[11]=31;

  int MonthLeapLength[12];
  MonthLeapLength[0]=31;
  MonthLeapLength[1]=29;
  MonthLeapLength[2]=31;
  MonthLeapLength[3]=30;
  MonthLeapLength[4]=31;
  MonthLeapLength[5]=30;
  MonthLeapLength[6]=31;
  MonthLeapLength[7]=31;
  MonthLeapLength[8]=30;
  MonthLeapLength[9]=31;
  MonthLeapLength[10]=30;
  MonthLeapLength[11]=31;

  return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}


INLINE
int monthOffsetKernelGpu(int m, bool leapYear) 
{
  int MonthOffset[13];
  MonthOffset[0]=0;
  MonthOffset[1]=31;
  MonthOffset[2]=59;
  MonthOffset[3]=90;
  MonthOffset[4]=120;
  MonthOffset[5]=151;
  MonthOffset[6]=181;
  MonthOffset[7]=212;
  MonthOffset[8]=243;
  MonthOffset[9]=273;
  MonthOffset[10]=304;
  MonthOffset[11]=334;
  MonthOffset[12]=365;

  int MonthLeapOffset[13];
  MonthLeapOffset[0]=0;
  MonthLeapOffset[1]=31;
  MonthLeapOffset[2]=60;
  MonthLeapOffset[3]=91;
  MonthLeapOffset[4]=121;
  MonthLeapOffset[5]=152;
  MonthLeapOffset[6]=182;
  MonthLeapOffset[7]=213;
  MonthLeapOffset[8]=244;
  MonthLeapOffset[9]=274;
  MonthLeapOffset[10]=305;
  MonthLeapOffset[11]=335;
  MonthLeapOffset[12]=366;

  return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}


INLINE
int yearOffsetKernelGpu(int y)
{

  int YearOffset[121];
  YearOffset[0] = 0;;
  YearOffset[1] = 366;;
  YearOffset[2] = 731;
  YearOffset[3] = 1096;
  YearOffset[4] = 1461;
  YearOffset[5] = 1827;
  YearOffset[6] = 2192;
  YearOffset[7] = 2557;
  YearOffset[8] = 2922;
  YearOffset[9] = 3288;
  YearOffset[10] = 3653;
  YearOffset[11] = 4018;
  YearOffset[12] = 4383;
  YearOffset[13] = 4749;
  YearOffset[14] = 5114;
  YearOffset[15] = 5479;
  YearOffset[16] = 5844;
  YearOffset[17] = 6210;
  YearOffset[18] = 6575;
  YearOffset[19] = 6940;
  YearOffset[20] = 7305;
  YearOffset[21] = 7671;


  YearOffset[22] = 8036;
  YearOffset[23] = 8401;
  YearOffset[24] = 8766;
  YearOffset[25] = 9132;
  YearOffset[26] = 9497;
  YearOffset[27] = 9862;
  YearOffset[28] = 10227;
  YearOffset[29] = 10593;
  YearOffset[30] = 10958;
  YearOffset[31] = 11323;
  YearOffset[32] = 11688;
  YearOffset[33] = 12054;

  YearOffset[34] = 12419;
  YearOffset[35] = 12784;
  YearOffset[36] = 13149;
  YearOffset[37] = 13515;
  YearOffset[38] = 13880;
  YearOffset[39] = 14245;
  YearOffset[40] = 14610;
  YearOffset[41] = 14976;
  YearOffset[42] = 15341;
  YearOffset[43] = 15706;
  YearOffset[44] = 16071;
  YearOffset[45] = 16437;

  YearOffset[46] = 16802;
  YearOffset[47] = 17167;
  YearOffset[48] = 17532;
  YearOffset[49] = 17898;
  YearOffset[50] = 18263;
  YearOffset[51] = 18628;
  YearOffset[52] = 18993;
  YearOffset[53] = 19359;
  YearOffset[54] = 19724;
  YearOffset[55] = 20089;
  YearOffset[56] = 20454;
  YearOffset[57] = 20820;

  YearOffset[58] = 21185;
  YearOffset[59] = 21550;
  YearOffset[60] = 21915;
  YearOffset[61] = 22281;
  YearOffset[62] = 22646;
  YearOffset[63] = 23011;
  YearOffset[64] = 23376;
  YearOffset[65] = 23742;
  YearOffset[66] = 24107;
  YearOffset[67] = 24472;
  YearOffset[68] = 24837;
  YearOffset[69] = 25203;

  YearOffset[70] = 25568;
  YearOffset[71] = 25933;
  YearOffset[72] = 26298;
  YearOffset[73] = 26664;
  YearOffset[74] = 27029;
  YearOffset[75] = 27394;
  YearOffset[76] = 27759;
  YearOffset[77] = 28125;
  YearOffset[78] = 28490;
  YearOffset[79] = 28855;
  YearOffset[80] = 29220;
  YearOffset[81] = 29586;

  YearOffset[82] = 29951;
  YearOffset[83] = 30316;
  YearOffset[84] = 30681;
  YearOffset[85] = 31047;
  YearOffset[86] = 31412;
  YearOffset[87] = 31777;
  YearOffset[88] = 32142;
  YearOffset[89] = 32508;
  YearOffset[90] = 32873;
  YearOffset[91] = 33238;
  YearOffset[92] = 33603;
  YearOffset[93] = 33969;

  YearOffset[94] = 34334;
  YearOffset[95] = 34699;
  YearOffset[96] = 35064;
  YearOffset[97] = 35430;
  YearOffset[98] = 35795;
  YearOffset[99] = 36160;
  YearOffset[100] = 36525;
  YearOffset[101] = 36891;
  YearOffset[102] = 37256;
  YearOffset[103] = 37621;
  YearOffset[104] = 37986;
  YearOffset[105] = 38352;

  YearOffset[106] = 38717;
  YearOffset[107] = 39082;
  YearOffset[108] = 39447;
  YearOffset[109] = 39813;
  YearOffset[110] = 40178;
  YearOffset[111] = 40543;
  YearOffset[112] = 40908;
  YearOffset[113] = 41274;
  YearOffset[114] = 41639;
  YearOffset[115] = 42004;
  YearOffset[116] = 42369;
  YearOffset[117] = 42735;
  YearOffset[118] = 43100;
  YearOffset[119] = 42735;
  YearOffset[120] = 43830;

  return YearOffset[y-1900];
}


INLINE
bool isLeapKernelGpu(int y) 
{
  bool YearIsLeap[121];

  YearIsLeap[0] = 1;;
  YearIsLeap[1] = 0;;
  YearIsLeap[2] = 0;
  YearIsLeap[3] = 0;//1096;
  YearIsLeap[4] = 1;//1461;
  YearIsLeap[5] = 0;//1827;
  YearIsLeap[6] = 0;//2192;
  YearIsLeap[7] = 0;//2557;
  YearIsLeap[8] = 1;//2922;
  YearIsLeap[9] = 0;//3288;
  YearIsLeap[10] = 0;//3653;
  YearIsLeap[11] = 0;//4018;
  YearIsLeap[12] = 1;//4383;
  YearIsLeap[13] = 0;//4749;
  YearIsLeap[14] = 0;//5114;
  YearIsLeap[15] = 0;//5479;
  YearIsLeap[16] = 1;//5844;
  YearIsLeap[17] = 0;//6210;
  YearIsLeap[18] = 0;//6575;
  YearIsLeap[19] = 0;//6940;
  YearIsLeap[20] = 1;//7305;
  YearIsLeap[21] = 0;//7671;
  YearIsLeap[22] = 0;//8036;
  YearIsLeap[23] = 0;//8401;
  YearIsLeap[24] = 1;//8766;
  YearIsLeap[25] = 0;//9132;
  YearIsLeap[26] = 0;//9497;
  YearIsLeap[27] = 0;//9862;
  YearIsLeap[28] = 1;//10227;
  YearIsLeap[29] = 0;//10593;
  YearIsLeap[30] = 0;//10958;
  YearIsLeap[31] = 0;//11323;
  YearIsLeap[32] = 1;//11688;
  YearIsLeap[33] = 0;//12054;
  YearIsLeap[34] = 0;//12419;
  YearIsLeap[35] = 0;//12784;
  YearIsLeap[36] = 1;//13149;
  YearIsLeap[37] = 0;//13515;
  YearIsLeap[38] = 0;//13880;
  YearIsLeap[39] = 0;//14245;
  YearIsLeap[40] = 1;//14610;
  YearIsLeap[41] = 0;//14976;
  YearIsLeap[42] = 0;//15341;
  YearIsLeap[43] = 0;//15706;
  YearIsLeap[44] = 1;//16071;
  YearIsLeap[45] = 0;//16437;
  YearIsLeap[46] = 0;//16802;
  YearIsLeap[47] = 0;//17167;
  YearIsLeap[48] = 1;//17532;
  YearIsLeap[49] = 0;//17898;
  YearIsLeap[50] = 0;//18263;
  YearIsLeap[51] = 0;//18628;
  YearIsLeap[52] = 1;//18993;
  YearIsLeap[53] = 0;//19359;
  YearIsLeap[54] = 0;//19724;
  YearIsLeap[55] = 0;//20089;
  YearIsLeap[56] = 1;//20454;
  YearIsLeap[57] = 0;//20820;
  YearIsLeap[58] = 0;//21185;
  YearIsLeap[59] = 0;//21550;
  YearIsLeap[60] = 1;//21915;
  YearIsLeap[61] = 0;//22281;
  YearIsLeap[62] = 0;//22646;
  YearIsLeap[63] = 0;//23011;
  YearIsLeap[64] = 1;//23376;
  YearIsLeap[65] = 0;//23742;
  YearIsLeap[66] = 0;//24107;
  YearIsLeap[67] = 0;//24472;
  YearIsLeap[68] = 1;//24837;
  YearIsLeap[69] = 0;//25203;
  YearIsLeap[70] = 0;//25568;
  YearIsLeap[71] = 0;//25933;
  YearIsLeap[72] = 1;//26298;
  YearIsLeap[73] = 0;//26664;
  YearIsLeap[74] = 0;//27029;
  YearIsLeap[75] = 0;//27394;
  YearIsLeap[76] = 1;//27759;
  YearIsLeap[77] = 0;//28125;
  YearIsLeap[78] = 0;//28490;
  YearIsLeap[79] = 0;//28855;
  YearIsLeap[80] = 1;//29220;
  YearIsLeap[81] = 0;//29586;
  YearIsLeap[82] = 0;//29951;
  YearIsLeap[83] = 0;//30316;
  YearIsLeap[84] = 1;//30681;
  YearIsLeap[85] = 0;//31047;
  YearIsLeap[86] = 0;//31412;
  YearIsLeap[87] = 0;//31777;
  YearIsLeap[88] = 1;//32142;
  YearIsLeap[89] = 0;//32508;
  YearIsLeap[90] = 0;//32873;
  YearIsLeap[91] = 0;//33238;
  YearIsLeap[92] = 1;//33603;
  YearIsLeap[93] = 0;//33969;
  YearIsLeap[94] = 0;//34334;
  YearIsLeap[95] = 0;//34699;
  YearIsLeap[96] = 1;//35064;
  YearIsLeap[97] = 0;//35430;
  YearIsLeap[98] = 0;//35795;
  YearIsLeap[99] = 0;//36160;
  YearIsLeap[100] = 1;// 36525;
  YearIsLeap[101] = 0;// 36891;
  YearIsLeap[102] = 0;// 37256;
  YearIsLeap[103] = 0;// 37621;
  YearIsLeap[104] = 1;// 37986;
  YearIsLeap[105] = 0;// 38352;
  YearIsLeap[106] = 0;//38717;
  YearIsLeap[107] = 0;//39082;
  YearIsLeap[108] = 1;//39447;
  YearIsLeap[109] = 0;//39813;
  YearIsLeap[110] = 0;//40178;
  YearIsLeap[111] = 0;//40543;
  YearIsLeap[112] = 1;//40908;
  YearIsLeap[113] = 0;//41274;
  YearIsLeap[114] = 0;//41639;
  YearIsLeap[115] = 0;//42004;
  YearIsLeap[116] = 1;//42369;
  YearIsLeap[117] = 0;//42735;
  YearIsLeap[118] = 0;//43100;
  YearIsLeap[119] = 0;//42735;
  YearIsLeap[120] = 1;//43830;

  return YearIsLeap[y-1900];
}


INLINE
bondsDateStruct intializeDateKernelGpu(int d, int m, int y) 
{
  bondsDateStruct currDate;

  currDate.day = d;
  currDate.month = m;
  currDate.year = y;

  bool leap = isLeapKernelGpu(y);
  int offset = monthOffsetKernelGpu(m,leap);

  currDate.dateSerialNum = d + offset + yearOffsetKernelGpu(y);

  return currDate;
}


INLINE
dataType yearFractionGpu(bondsDateStruct d1,
    bondsDateStruct d2, int dayCounter)
{
  return dayCountGpu(d1, d2, dayCounter) / (dataType)360.0; 
}


INLINE
int dayCountGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter) 
{
  if (dayCounter == USE_EXACT_DAY)
  {
    int dd1 = d1.day, dd2 = d2.day;
    int mm1 = d1.month, mm2 = d2.month;
    int yy1 = d1.year, yy2 = d2.year;

    if (dd2 == 31 && dd1 < 30) 
    { 
      dd2 = 1; mm2++; 
    }

    return 360*(yy2-yy1) + 30*(mm2-mm1-1) + MAX(0, 30-dd1) + MIN(30, dd2);
  }
  else
  {
    return (d2.dateSerialNum - d1.dateSerialNum);
  }
}


INLINE
dataType couponNotionalGpu()
{
  return (dataType)100.0;
}


INLINE
dataType bondNotionalGpu()
{
  return (dataType)100.0;
}



INLINE
dataType fixedRateCouponNominalGpu()
{
  return (dataType)100.0;
}


INLINE
bool eventHasOccurredGpu(bondsDateStruct currDate, bondsDateStruct eventDate)
{
  return eventDate.dateSerialNum > currDate.dateSerialNum;
}


INLINE
bool cashFlowHasOccurredGpu(bondsDateStruct refDate, bondsDateStruct eventDate)
{
  return eventHasOccurredGpu(refDate, eventDate);
}


INLINE
bondsDateStruct advanceDateGpu(bondsDateStruct date, int numMonthsAdvance) 
{
  int d = date.day;
  int m = date.month+numMonthsAdvance;
  int y = date.year;

  while (m > 12) 
  {
    m -= 12;
    y += 1;
  }

  while (m < 1) 
  {
    m += 12;
    y -= 1;
  }

  int length = monthLengthKernelGpu(m, isLeapKernelGpu(y));
  if (d > length)
    d = length;

  bondsDateStruct newDate = intializeDateKernelGpu(d, m, y);

  return newDate;
}


INLINE
dataType getDirtyPriceGpu(bondStruct *bond, 
    bondsYieldTermStruct *discountCurve, 
    bondsDateStruct *currDate, 
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();
  bondsDateStruct currentDate = currDate[bondNum];

  if (currentDate.dateSerialNum < bond[bondNum].startDate.dateSerialNum)
  {
    currentDate = bond[bondNum].startDate;
  }

  return cashFlowsNpvGpu(cashFlows,
      discountCurve[bondNum], false, currentDate, currentDate, numLegs) * (dataType)100.0 / currentNotional;
}


INLINE
dataType getAccruedAmountGpu(bondsDateStruct *maturityDate, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();
  if (currentNotional == (dataType)0.0)
    return (dataType)0.0;

  return cashFlowsAccruedAmountGpu(cashFlows, false, date, numLegs, maturityDate, bondNum) * 
    (dataType)100.0 / bondNotionalGpu();
}


INLINE
dataType bondFunctionsAccruedAmountGpu(bondsDateStruct *maturityDate, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs) 
{
  return cashFlowsAccruedAmountGpu(cashFlows, false, date, numLegs, maturityDate, bondNum) * 
    (dataType)100.0 / bondNotionalGpu();
}


INLINE
dataType cashFlowsAccruedAmountGpu(cashFlowsStruct cashFlows,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    int numLegs, bondsDateStruct* maturityDate, int bondNum) 
{
  int legComputeNum = cashFlowsNextCashFlowNumGpu(cashFlows, currDate, numLegs); 

  dataType result = 0.0;

  for (int i = legComputeNum; i < (numLegs); ++i)
  {
    result += fixedRateCouponAccruedAmountGpu(cashFlows, i, currDate, maturityDate, bondNum);
  }

  return result;
}


INLINE
dataType fixedRateCouponAccruedAmountGpu(cashFlowsStruct cashFlows, int numLeg, 
    bondsDateStruct d, bondsDateStruct* maturityDate, int bondNum) 
{
  if (d.dateSerialNum <= cashFlows.legs[numLeg].accrualStartDate.dateSerialNum || 
      d.dateSerialNum > maturityDate[bondNum].dateSerialNum) 
  {
    return (dataType)0.0;
  }
  else
  {
    bondsDateStruct endDate = cashFlows.legs[numLeg].accrualEndDate;
    if (d.dateSerialNum < cashFlows.legs[numLeg].accrualEndDate.dateSerialNum)
    {
      endDate = d;
    }

    return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate, 
          cashFlows.legs[numLeg].accrualStartDate, endDate, cashFlows.dayCounter) - (dataType)1.0);
  }
}


INLINE
dataType cashFlowsNpvGpu(cashFlowsStruct cashFlows,
    bondsYieldTermStruct discountCurve,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs) 
{
  npvDate = currDate;

  dataType totalNPV = 0.0;

  int i;

  for (i=0; i<numLegs; ++i) {
    if (!(cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate)))
      totalNPV += fixedRateCouponAmountGpu(cashFlows, i) *
        bondsYieldTermStructureDiscountGpu(discountCurve, cashFlows.legs[i].paymentDate);
  }

  return totalNPV/bondsYieldTermStructureDiscountGpu(discountCurve, npvDate);
}



INLINE
dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct, bondsDateStruct t)
{
  ytStruct.intRate.rate = ytStruct.forward;
  ytStruct.intRate.freq = ytStruct.frequency;
  ytStruct.intRate.comp = ytStruct.compounding;
  return flatForwardDiscountImplGpu(ytStruct.intRate, yearFractionGpu(ytStruct.refDate, t, ytStruct.dayCounter));
}



INLINE
dataType flatForwardDiscountImplGpu(intRateStruct intRate, dataType t) 
{
  return interestRateDiscountFactorGpu(intRate, t);
}


INLINE
dataType interestRateDiscountFactorGpu(intRateStruct intRate, dataType t) 
{
  return (dataType)1.0/interestRateCompoundFactorGpuTwoArgs(intRate, t);
}


INLINE
dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t) 
{
  if (intRate.comp == SIMPLE_INTEREST)
    return (dataType)1.0 + intRate.rate*t;
  else if (intRate.comp == COMPOUNDED_INTEREST)
    return sycl::pow((dataType)1.0+intRate.rate/intRate.freq, intRate.freq*t);
  else if (intRate.comp == CONTINUOUS_INTEREST)
    return sycl::exp(intRate.rate*t);
  return (dataType)0.0;
}


INLINE
dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg) 
{
  if (cashFlows.legs[numLeg].amount == COMPUTE_AMOUNT)
  {
    return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate, cashFlows.legs[numLeg].accrualStartDate,
          cashFlows.legs[numLeg].accrualEndDate, cashFlows.dayCounter) - (dataType)1.0);
  }
  else
  {
    return cashFlows.legs[numLeg].amount;
  }
}

INLINE
dataType interestRateCompoundFactorGpu(intRateStruct intRate, bondsDateStruct d1,
    bondsDateStruct d2, int dayCounter)
{
  dataType t = yearFractionGpu(d1, d2, dayCounter);
  return interestRateCompoundFactorGpuTwoArgs(intRate, t);
}



INLINE
dataType interestRateImpliedRateGpu(dataType compound, int comp, dataType freq, dataType t) 
{
  dataType r = 0.0f;
  if (compound==(dataType)1.0) 
  {
    r = 0.0;
  } 
  else 
  {
    switch (comp) 
    {
      case SIMPLE_INTEREST:
        r = (compound - (dataType)1.0)/t;
        break;
      case COMPOUNDED_INTEREST:
        r = (sycl::pow((dataType)compound, (dataType)1.0/((freq)*t))-(dataType)1.0)*(freq);
        break;
    }
  }

  return r;
}



INLINE
int cashFlowsNextCashFlowNumGpu(cashFlowsStruct cashFlows,
    bondsDateStruct currDate,
    int numLegs) 
{
  int i;
  for (i = 0; i < numLegs; ++i) 
  {
    if ( ! (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate) ))
      return i;
  }

  return (numLegs-1);
}


INLINE
dataType getBondYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondStruct *bond,
    bondsDateStruct *maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType currentNotional = bondNotionalGpu();

  if (currentNotional == (dataType)0.0)
    return (dataType)0.0;

  if (bond[bondNum].startDate.dateSerialNum > settlement.dateSerialNum)
  {
    settlement = bond[bondNum].startDate;
  }

  return getBondFunctionsYieldGpu(cleanPrice, dc, comp, freq,
      settlement, accuracy, maxEvaluations,
      maturityDate, bondNum, cashFlows, numLegs);
}


INLINE
dataType getBondFunctionsYieldGpu(dataType cleanPrice,
    int dc,
    int comp,
    dataType freq,
    bondsDateStruct settlement,
    dataType accuracy,
    int maxEvaluations,
    bondsDateStruct* maturityDate,
    int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
  dataType dirtyPrice = cleanPrice + bondFunctionsAccruedAmountGpu(maturityDate, settlement, bondNum, cashFlows, numLegs); 
  dirtyPrice /= (dataType)100.0 / bondNotionalGpu();

  return getCashFlowsYieldGpu(cashFlows, dirtyPrice,
      dc, comp, freq,
      false, settlement, settlement, numLegs,
      accuracy, maxEvaluations, (dataType)0.05);
}



INLINE
dataType getCashFlowsYieldGpu(cashFlowsStruct leg,
    dataType npv,
    int dayCounter,
    int compounding,
    dataType frequency,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs,
    dataType accuracy,
    int maxIterations,
    dataType guess)
{
  //Brent solver;
  solverStruct solver;
  solver.maxEvaluations_ = maxIterations;
  irrFinderStruct objFunction;

  objFunction.npv = npv;
  objFunction.dayCounter = dayCounter;
  objFunction.comp = compounding;
  objFunction.freq = frequency;
  objFunction.includecurrDateFlows = includecurrDateFlows;
  objFunction.currDate = currDate;
  objFunction.npvDate = npvDate;

  return solverSolveGpu(solver, objFunction, accuracy, guess, guess/(dataType)10.0, leg, numLegs);
}


INLINE
dataType solverSolveGpu(solverStruct solver,
    irrFinderStruct f,
    dataType accuracy,
    dataType guess,
    dataType step,
    cashFlowsStruct cashFlows,
    int numLegs)
{
  // check whether we really want to use epsilon
  accuracy = MAX(accuracy, QL_EPSILON_GPU);

  dataType growthFactor = (dataType)1.6;
  int flipflop = -1;

  solver.root_ = guess;
  solver.fxMax_ = fOpGpu(f, solver.root_, cashFlows, numLegs);

  // monotonically crescent bias, as in optionValue(volatility)
  if (closeGpu(solver.fxMax_,(dataType)0.0))
  {
    return solver.root_;
  }
  else if (closeGpu(solver.fxMax_, (dataType)0.0)) 
  {
    solver.xMin_ = /*enforceBounds*/(solver.root_ - step);
    solver.fxMin_ = fOpGpu(f, solver.xMin_, cashFlows, numLegs);
    solver.xMax_ = solver.root_;
  } 
  else 
  {
    solver.xMin_ = solver.root_;
    solver.fxMin_ = solver.fxMax_;
    solver.xMax_ = /*enforceBounds*/(solver.root_+step);
    solver.fxMax_ = fOpGpu(f, solver.xMax_, cashFlows, numLegs);
  }

  solver.evaluationNumber_ = 2;
  while (solver.evaluationNumber_ <= solver.maxEvaluations_) 
  {
    if (solver.fxMin_*solver.fxMax_ <= (dataType)0.0) 
    {
      if (closeGpu(solver.fxMin_, (dataType)0.0))
        return solver.xMin_;
      if (closeGpu(solver.fxMax_, (dataType)0.0))
        return solver.xMax_;
      solver.root_ = (solver.xMax_+solver.xMin_)/(dataType)2.0;
      return solveImplGpu(solver, f, accuracy, cashFlows, numLegs);
    }
    if (sycl::fabs(solver.fxMin_) < sycl::fabs(solver.fxMax_)) 
    {
      solver.xMin_ = /*enforceBounds*/(solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
      solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
    } 
    else if (sycl::fabs(solver.fxMin_) > sycl::fabs(solver.fxMax_)) 
    {
      solver.xMax_ = /*enforceBounds*/(solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
      solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
    } 
    else if (flipflop == -1) 
    {
      solver.xMin_ = /*enforceBounds*/(solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
      solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
      solver.evaluationNumber_++;
      flipflop = 1;
    } 
    else if (flipflop == 1) 
    {
      solver.xMax_ = /*enforceBounds*/(solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
      solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
      flipflop = -1;
    }
    solver.evaluationNumber_++;
  }

  return (dataType)0.0;
}


INLINE
dataType cashFlowsNpvYieldGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs) 
{
  dataType npv = 0.0;
  dataType discount = 1.0;
  bondsDateStruct lastDate;
  bool first = true;

  int i;
  for (i=0; i<numLegs; ++i) 
  {
    if (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate))
      continue;

    bondsDateStruct couponDate = cashFlows.legs[i].paymentDate;
    dataType amount = fixedRateCouponAmountGpu(cashFlows, i);
    if (first) 
    {
      first = false;
      if (i > 0) {
        lastDate = advanceDateGpu(cashFlows.legs[i].paymentDate, -1*6); 
      } else {
        lastDate = cashFlows.legs[i].accrualStartDate;
      }
      discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(npvDate, couponDate, y.dayCounter));
    } 
    else  
    {
      discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(lastDate, couponDate, y.dayCounter));
    }

    lastDate = couponDate;

    npv += amount * discount;
  }

  return npv;
}


INLINE
dataType fOpGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
  intRateStruct yield;

  yield.rate = y;
  yield.comp = f.comp;
  yield.freq = f.freq;
  yield.dayCounter = f.dayCounter;

  dataType NPV = cashFlowsNpvYieldGpu(cashFlows,
      yield,
      false,
      f.currDate,
      f.npvDate, numLegs);

  return (f.npv - NPV);
}



INLINE
dataType fDerivativeGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
  intRateStruct yield;
  yield.rate = y;
  yield.dayCounter = f.dayCounter;
  yield.comp = f.comp;
  yield.freq = f.freq;

  return modifiedDurationGpu(cashFlows, yield,
      f.includecurrDateFlows,
      f.currDate, f.npvDate, numLegs);
}


INLINE
bool closeGpu(dataType x, dataType y)
{
  return closeGpuThreeArgs(x,y,42);
}


INLINE
bool closeGpuThreeArgs(dataType x, dataType y, int n)
{
  dataType diff = sycl::fabs(x-y);
  dataType tolerance = n*QL_EPSILON_GPU;

  return diff <= tolerance*sycl::fabs(x) &&
    diff <= tolerance*sycl::fabs(y);
}


INLINE
dataType solveImplGpu(solverStruct solver, irrFinderStruct f,
    dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs)
{
  dataType froot, dfroot, dx, dxold;
  dataType xh, xl;

  // Orient the search so that f(xl) < 0
  if (solver.fxMin_ < (dataType)0.0) 
  {
    xl = solver.xMin_;
    xh = solver.xMax_;
  } 
  else 
  {
    xh = solver.xMin_;
    xl = solver.xMax_;
  }

  // the "stepsize before last"
  dxold = solver.xMax_ - solver.xMin_;
  // it was dxold=std::sycl::fabs(xMax_-xMin_); in Numerical Recipes
  // here (xMax_-xMin_ > 0) is verified in the constructor

  // and the last step
  dx = dxold;

  froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
  dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);

  ++solver.evaluationNumber_;

  while (solver.evaluationNumber_<=solver.maxEvaluations_) 
  {
    // Bisect if (out of range || not decreasing fast enough)
    if ((((solver.root_-xh)*dfroot-froot)*
          ((solver.root_-xl)*dfroot-froot) > (dataType)0.0)
        || (sycl::fabs((dataType)2.0*froot) > sycl::fabs(dxold*dfroot))) 
    {
      dxold = dx;
      dx = (xh-xl)/(dataType)2.0;
      solver.root_=xl+dx;
    } 
    else 
    {
      dxold = dx;
      dx = froot/dfroot;
      solver.root_ -= dx;
    }

    // Convergence criterion
    if (sycl::fabs(dx) < xAccuracy)
      return solver.root_;
    froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
    dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);
    ++solver.evaluationNumber_;
    if (froot < (dataType)0.0)
      xl=solver.root_;
    else
      xh=solver.root_;
  }

  return solver.root_;
}


INLINE
dataType modifiedDurationGpu(cashFlowsStruct cashFlows,
    intRateStruct y,
    bool includecurrDateFlows,
    bondsDateStruct currDate,
    bondsDateStruct npvDate,
    int numLegs)
{
  dataType P = 0.0;
  dataType dPdy = 0.0;
  dataType r = y.rate;
  dataType N = y.freq;
  int dc = y.dayCounter;

  int i;
  for (i=0; i<numLegs; ++i) 
  {
    if (!cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate)) 
    {
      dataType t = yearFractionGpu(npvDate,
          cashFlows.legs[i].paymentDate, dc);
      dataType c = fixedRateCouponAmountGpu(cashFlows, i);  
      dataType B = interestRateDiscountFactorGpu(y, t); 

      P += c * B;
      if (y.comp == SIMPLE_INTEREST)
        dPdy -= c * B*B * t;
      if (y.comp == COMPOUNDED_INTEREST)
        dPdy -= c * t * B/(1+r/N);
      if (y.comp == CONTINUOUS_INTEREST)
        dPdy -= c * B * t;
      if (y.comp == SIMPLE_THEN_COMPOUNDED_INTEREST)
      {
        if (t<=(dataType)1.0/N)
          dPdy -= c * B*B * t;
        else
          dPdy -= c * t * B/((dataType)1+r/N);
      }
    }
  }

  if (P == (dataType)0.0) // no cashflows
  {
    return (dataType)0.0;
  }
  return (-1*dPdy)/P; // reverse derivative sign
}


__attribute__ ((always_inline))
void bonds (
    sycl::nd_item<1> &item,
    bondsYieldTermStruct* __restrict discountCurve,
    bondsYieldTermStruct* __restrict repoCurve,
    bondsDateStruct* __restrict currDate,
    bondsDateStruct* __restrict maturityDate,
    dataType* __restrict bondCleanPrice,
    bondStruct* __restrict bond,
    dataType* __restrict dummyStrike,
    dataType* __restrict dirtyPrice,
    dataType* __restrict accruedAmountCurrDate,
    dataType* __restrict cleanPrice,
    dataType* __restrict bondForwardVal,
    int n)
{
  int bondNum = item.get_global_id(0);
  if (bondNum < n)
  {
    int numLegs;

    int numCashFlows = 0;

    bondsDateStruct currCashflowDate = bond[bondNum].maturityDate;

    while (currCashflowDate.dateSerialNum > bond[bondNum].startDate.dateSerialNum)
    {
      numCashFlows++;
      currCashflowDate = advanceDateGpu(currCashflowDate, -6); 
    }

    numLegs = numCashFlows+1;

    cashFlowsStruct cashFlows; 
    couponStruct cashLegs[9];
    cashFlows.legs = cashLegs;

    cashFlows.intRate.dayCounter = USE_EXACT_DAY;
    cashFlows.intRate.rate  = bond[bondNum].rate;
    cashFlows.intRate.freq  = ANNUAL_FREQ;
    cashFlows.intRate.comp  = SIMPLE_INTEREST;
    cashFlows.dayCounter  = USE_EXACT_DAY;
    cashFlows.nominal  = (dataType)100.0;

    //bondsDateStruct currPaymentDate;
    bondsDateStruct currStartDate = advanceDateGpu(bond[bondNum].maturityDate, (numLegs - 1)*-6);
    bondsDateStruct currEndDate = advanceDateGpu(currStartDate, 6); 

    int cashFlowNum;
    for (cashFlowNum = 0; cashFlowNum < numLegs-1; cashFlowNum++)
    {
      cashFlows.legs[cashFlowNum].paymentDate = currEndDate;

      cashFlows.legs[cashFlowNum].accrualStartDate  = currStartDate;
      cashFlows.legs[cashFlowNum].accrualEndDate  = currEndDate;

      cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

      currStartDate = currEndDate;
      currEndDate = advanceDateGpu(currEndDate, 6); 
    }

    cashFlows.legs[numLegs-1].paymentDate  = bond[bondNum].maturityDate;
    cashFlows.legs[numLegs-1].accrualStartDate = currDate[bondNum];
    cashFlows.legs[numLegs-1].accrualEndDate  = currDate[bondNum];
    cashFlows.legs[numLegs-1].amount = (dataType)100.0;

    bondForwardVal[bondNum] = getBondYieldGpu(bondCleanPrice[bondNum],
        USE_EXACT_DAY,
        COMPOUNDED_INTEREST,
        (dataType)2.0,
        currDate[bondNum],
        ACCURACY,
        100,
        bond, 
        maturityDate,
        bondNum, cashFlows, numLegs);

    discountCurve[bondNum].forward = bondForwardVal[bondNum];
    dirtyPrice[bondNum] = getDirtyPriceGpu(bond, discountCurve, currDate, bondNum, cashFlows, numLegs);
    accruedAmountCurrDate[bondNum] = getAccruedAmountGpu(maturityDate, currDate[bondNum], bondNum, cashFlows, numLegs);

    cleanPrice[bondNum] = dirtyPrice[bondNum] - accruedAmountCurrDate[bondNum];
  }
}

long getBondsResultsGpu(sycl::queue &q, inArgsStruct inArgsHost, resultsStruct resultsFromGpu, int numBonds)
{
  bondsYieldTermStruct *discountCurveGpu = sycl::malloc_device<bondsYieldTermStruct>(numBonds, q);
  bondsYieldTermStruct *repoCurveGpu = sycl::malloc_device<bondsYieldTermStruct>(numBonds, q);
  bondsDateStruct *currDateGpu = sycl::malloc_device<bondsDateStruct>(numBonds, q);
  bondsDateStruct *maturityDateGpu = sycl::malloc_device<bondsDateStruct>(numBonds, q);
  dataType *bondCleanPriceGpu = sycl::malloc_device<dataType>(numBonds, q);
  bondStruct *bondGpu = sycl::malloc_device<bondStruct>(numBonds, q);
  dataType *dummyStrikeGpu = sycl::malloc_device<dataType>(numBonds, q);
  dataType *dirtyPriceGpu = sycl::malloc_device<dataType>(numBonds, q);
  dataType *accruedAmountCurrDateGpu = sycl::malloc_device<dataType>(numBonds, q);
  dataType *cleanPriceGpu = sycl::malloc_device<dataType>(numBonds, q);
  dataType *bondForwardValGpu = sycl::malloc_device<dataType>(numBonds, q);

  q.memcpy(discountCurveGpu, inArgsHost.discountCurve, sizeof(bondsYieldTermStruct) * numBonds);
  q.memcpy(repoCurveGpu, inArgsHost.repoCurve, numBonds*sizeof(bondsYieldTermStruct));
  q.memcpy(currDateGpu, inArgsHost.currDate, numBonds*sizeof(bondsDateStruct));
  q.memcpy(maturityDateGpu, inArgsHost.maturityDate, numBonds*sizeof(bondsDateStruct));
  q.memcpy(bondCleanPriceGpu, inArgsHost.bondCleanPrice, numBonds*sizeof(dataType));
  q.memcpy(bondGpu, inArgsHost.bond, numBonds*sizeof(bondStruct));
  q.memcpy(dummyStrikeGpu, inArgsHost.dummyStrike, numBonds*sizeof(dataType));
  q.wait();

  sycl::range<1> gws ((numBonds + 255)/256*256);
  sycl::range<1> lws (256);

  struct timeval start;
  struct timeval end;
  gettimeofday(&start, NULL);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class kernel>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      bonds(item,
            discountCurveGpu,
            repoCurveGpu,
            currDateGpu,
            maturityDateGpu,
            bondCleanPriceGpu,
            bondGpu,
            dummyStrikeGpu,
            dirtyPriceGpu,
            accruedAmountCurrDateGpu,
            cleanPriceGpu,
            bondForwardValGpu,
            numBonds);
    });
  }).wait();

  gettimeofday(&end, NULL);
  long ktime = (end.tv_sec - start.tv_sec) * 1e6 + end.tv_usec - start.tv_usec;

  q.memcpy(resultsFromGpu.dirtyPrice, dirtyPriceGpu, numBonds*sizeof(dataType));
  q.memcpy(resultsFromGpu.accruedAmountCurrDate, accruedAmountCurrDateGpu, numBonds*sizeof(dataType));
  q.memcpy(resultsFromGpu.cleanPrice, cleanPriceGpu, numBonds*sizeof(dataType));
  q.memcpy(resultsFromGpu.bondForwardVal, bondForwardValGpu, numBonds*sizeof(dataType));
  q.wait();

  sycl::free(discountCurveGpu, q);
  sycl::free(repoCurveGpu, q);
  sycl::free(currDateGpu, q);
  sycl::free(maturityDateGpu, q);
  sycl::free(bondCleanPriceGpu, q);
  sycl::free(bondGpu, q);
  sycl::free(dummyStrikeGpu, q);

  sycl::free(dirtyPriceGpu, q);
  sycl::free(accruedAmountCurrDateGpu, q);
  sycl::free(cleanPriceGpu, q);
  sycl::free(bondForwardValGpu, q);
  return ktime;
}
