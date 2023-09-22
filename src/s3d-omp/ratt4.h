    const real TEMP = T[i]*tconv;
    //real ALOGT = LOG(TEMP);
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(5)*EG(16)));
    RB(51) = RF(51) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(2)*EG(2)*EG(15)*PFAC));
    RB(52) = RF(52) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(2)*EG(17)));
    RB(53) = RF(53) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(6)*EG(9)));
    RB(54) = RF(54) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(10)), (EG(5)*EG(17)));
    RB(55) = RF(55) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(14)*PFAC), EG(26));
    RB(56) = RF(56) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(10)), (EG(2)*EG(19)));
    RB(57) = RF(57) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(10)), (EG(1)*EG(19)));
    RB(58) = RF(58) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(59) = RF(59) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(11)), (EG(1)*EG(9)));
    RB(60) = RF(60) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(1)*EG(14)));
    RB(61) = RF(61) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(2)*EG(16)));
    RB(62) = RF(62) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(11)), (EG(2)*EG(17)));
    RB(63) = RF(63) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(11)), (EG(2)*EG(12)));
    RB(64) = RF(64) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(2)*EG(5)*EG(14)*PFAC));
    RB(65) = RF(65) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(6)*EG(14)));
    RB(66) = RF(66) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(67) = RF(67) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(68) = RF(68) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(69) = RF(69) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(15)), (EG(14)*EG(17)));
    RB(70) = RF(70) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)*PFAC), EG(18));
    RB(71) = RF(71) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)), (EG(1)*EG(16)));
    RB(72) = RF(72) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(17)), (EG(5)*EG(16)));
    RB(73) = RF(73) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(17)), (EG(6)*EG(16)));
    RB(74) = RF(74) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(17)), (EG(7)*EG(16)));
    RB(75) = RF(75) * MIN(rtemp_inv, SMALL_INV);

