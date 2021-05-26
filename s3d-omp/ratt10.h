
    const real TEMP = T[i]*tconv;
    const real ALOGT = LOG(TEMP);

    RKLOW(1) = EXP((real)4.22794408e1  -(real)9.e-1*ALOGT   + DIV((real)8.55468335e2,TEMP));
    RKLOW(2) = EXP((real)6.37931383e1  -(real)3.42e0*ALOGT  - DIV((real)4.24463259e4,TEMP));
    RKLOW(3) = EXP((real)6.54619238e1  -(real)3.74e0*ALOGT  - DIV((real)9.74227469e2,TEMP));
    RKLOW(4) = EXP((real)5.55621468e1  -(real)2.57e0*ALOGT  - DIV((real)7.17083751e2,TEMP));
    RKLOW(5) = EXP((real)6.33329483e1  -(real)3.14e0*ALOGT  - DIV((real)6.18956501e2,TEMP));
    RKLOW(6) = EXP((real)7.69748493e1  -(real)5.11e0*ALOGT  - DIV((real)3.57032226e3,TEMP));
    RKLOW(7) = EXP((real)6.98660102e1  -(real)4.8e0*ALOGT   - DIV((real)2.79788467e3,TEMP));
    RKLOW(8) = EXP((real)7.68923562e1  -(real)4.76e0*ALOGT  - DIV((real)1.22784867e3,TEMP));
    RKLOW(9) = EXP((real)1.11312542e2  -(real)9.588e0*ALOGT - DIV((real)2.566405e3,TEMP));
    RKLOW(10) = EXP((real)1.15700234e2 -(real)9.67e0*ALOGT  - DIV((real)3.13000767e3,TEMP));
    RKLOW(11) = EXP((real)3.54348644e1 -(real)6.4e-1*ALOGT  - DIV((real)2.50098684e4,TEMP));
    RKLOW(12) = EXP((real)6.3111756e1  -(real)3.4e0*ALOGT   - DIV((real)1.80145126e4,TEMP));
    RKLOW(13) = EXP((real)9.57409899e1 -(real)7.64e0*ALOGT  - DIV((real)5.98827834e3,TEMP));
    RKLOW(14) = EXP((real)6.9414025e1  -(real)3.86e0*ALOGT  - DIV((real)1.67067934e3,TEMP));
    RKLOW(15) = EXP((real)1.35001549e2 -(real)1.194e1*ALOGT - DIV((real)4.9163262e3,TEMP));
    RKLOW(16) = EXP((real)9.14494773e1 -(real)7.297e0*ALOGT - DIV((real)2.36511834e3,TEMP));
    RKLOW(17) = EXP((real)1.17075165e2 -(real)9.31e0*ALOGT  - DIV((real)5.02512164e4,TEMP));
    RKLOW(18) = EXP((real)9.68908955e1 -(real)7.62e0*ALOGT  - DIV((real)3.50742017e3,TEMP));
    RKLOW(19) = EXP((real)9.50941235e1 -(real)7.08e0*ALOGT  - DIV((real)3.36400342e3,TEMP));
    RKLOW(20) = EXP((real)1.38440285e2 -(real)1.2e1*ALOGT   - DIV((real)3.00309643e3,TEMP));
    RKLOW(21) = EXP((real)8.93324137e1 -(real)6.66e0*ALOGT  - DIV((real)3.52251667e3,TEMP));

