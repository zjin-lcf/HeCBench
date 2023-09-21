    const real TEMP = T[i]*tconv;
    //const real ALOGT = LOG((TEMP));
    real CTOT = (real)0.0;
    real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    real CPRLOG, FLOG, FC;
    real SQR;
    const real SMALL = FLT_MIN;

    #pragma unroll 22
    for (unsigned int k=1; k<=22; k++) {
        CTOT += C(k);
    }

    real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
    real CTB_114= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)1.5e0*C(14) + (real)1.5e0*C(15) ;
    real CTB_16 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15) ;

    //     If fall-off (pressure correction):

    PR = RKLOW(1) * DIV(CTB_16, RF(16));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.654e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.346e-1*EXP(DIV(-TEMP,(real)1.756e3))
    + EXP(DIV((real)-5.182e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(16) = RF(16) * PCOR;
    RB(16) = RB(16) * PCOR;

    PR = RKLOW(2) * DIV(CTB_10, RF(31));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)6.8e-2*EXP(DIV(-TEMP,(real)1.97e2)) + (real)9.32e-1*EXP(DIV(-TEMP,(real)1.54e3))
    + EXP(DIV((real)-1.03e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV (CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV (FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(31) = RF(31) * PCOR;
    RB(31) = RB(31) * PCOR;

    PR = RKLOW(3) * DIV(CTB_10, RF(39));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.243e-1*EXP(DIV(-TEMP,(real)2.37e2)) + (real)5.757e-1*EXP(DIV(-TEMP,(real)1.652e3))
    + EXP(DIV((real)-5.069e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(39) = RF(39) * PCOR;
    RB(39) = RB(39) * PCOR;

    PR = RKLOW(4) * DIV(CTB_10, RF(41));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.176e-1*EXP(DIV(-TEMP,(real)2.71e2)) + (real)7.824e-1*EXP(DIV(-TEMP,(real)2.755e3))
    + EXP(DIV((real)-6.57e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(41) = RF(41) * PCOR;
    RB(41) = RB(41) * PCOR;

    PR = RKLOW(5) * DIV(CTB_10, RF(48));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.2e-1*EXP(DIV(-TEMP,(real)7.8e1)) + (real)6.8e-1*EXP(DIV(-TEMP,(real)1.995e3))
    + EXP(DIV((real)-5.59e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(48) = RF(48) * PCOR;
    RB(48) = RB(48) * PCOR;

    PR = RKLOW(6) * DIV(CTB_10, RF(56));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.093e-1*EXP(DIV(-TEMP,(real)2.75e2)) + (real)5.907e-1*EXP(DIV(-TEMP,(real)1.226e3))
    + EXP(DIV((real)-5.185e3, TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(56) = RF(56) * PCOR;
    RB(56) = RB(56) * PCOR;

    PR = RKLOW(7) * DIV(CTB_10, RF(71));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.42e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.58e-1*EXP(DIV(-TEMP,(real)1.555e3))
    + EXP(DIV((real)-4.2e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(71) = RF(71) * PCOR;
    RB(71) = RB(71) * PCOR;

    PR = RKLOW(8) * DIV(CTB_10, RF(78));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.17e-1*EXP(DIV(-TEMP,(real)7.4e1)) + (real)7.83e-1*EXP(DIV(-TEMP,(real)2.941e3))
    + EXP(DIV((real)-6.964e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(78) = RF(78) * PCOR;
    RB(78) = RB(78) * PCOR;

    PR = RKLOW(9) * DIV(CTB_10, RF(89));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.827e-1*EXP(DIV(-TEMP,(real)1.3076e1)) + (real)6.173e-1*EXP(DIV(-TEMP,(real)2.078e3))
    + EXP(DIV((real)-5.093e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(89) = RF(89) * PCOR;
    RB(89) = RB(89) * PCOR;

    PR = RKLOW(10) * DIV(CTB_10, RF(93));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.675e-1*EXP(DIV(-TEMP,(real)1.51e2)) + (real)5.325e-1*EXP(DIV(-TEMP,(real)1.038e3))
    + EXP(DIV((real)-4.97e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(93) = RF(93) * PCOR;
    RB(93) = RB(93) * PCOR;

    PR = RKLOW(11) * DIV(CTB_114, RF(114));
    PCOR = DIV(PR, ((real)1.0 + PR));
    RF(114) = RF(114) * PCOR;
    RB(114) = RB(114) * PCOR;

    PR = RKLOW(12) * DIV(CTB_10, RF(115));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)-9.816e-1*EXP(DIV(-TEMP,(real)5.3837e3)) +
            (real)1.9816e0*EXP(DIV(-TEMP,(real)4.2932e0))  +
            EXP(DIV((real)7.95e-2,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(115) = RF(115) * PCOR;
    RB(115) = RB(115) * PCOR;
