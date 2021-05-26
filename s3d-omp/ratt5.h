    const real TEMP = T[i]*tconv;
    //const real ALOGT = LOG(TEMP);
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(7)*EG(17)), (EG(8)*EG(16)));
    RB(76) = RF(76) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(17)), (EG(2)*EG(26)));
    RB(77) = RF(77) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(12)*PFAC), EG(13));
    RB(78) = RF(78) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(12)), (EG(2)*EG(17)));
    RB(79) = RF(79) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(10)));
    RB(80) = RF(80) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(11)));
    RB(81) = RF(81) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(3)*EG(18)));
    RB(82) = RF(82) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(5)*EG(17)));
    RB(83) = RF(83) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(4)*EG(13)));
    RB(84) = RF(84) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(5)*EG(18)));
    RB(85) = RF(85) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(12)), (EG(7)*EG(13)));
    RB(86) = RF(86) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(12)), (EG(2)*EG(21)));
    RB(87) = RF(87) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)), (EG(13)*EG(14)));
    RB(88) = RF(88) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)*PFAC), EG(28));
    RB(89) = RF(89) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(17)), (EG(13)*EG(16)));
    RB(90) = RF(90) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(12)), (EG(2)*EG(22)));
    RB(91) = RF(91) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(12)), (EG(2)*EG(22)));
    RB(92) = RF(92) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)*PFAC), EG(24));
    RB(93) = RF(93) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)), (EG(2)*EG(23)));
    RB(94) = RF(94) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(25)), (EG(14)*EG(22)));
    RB(95) = RF(95) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(1)*EG(17)));
    RB(96) = RF(96) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(5)*EG(12)));
    RB(97) = RF(97) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(6)*EG(11)));
    RB(98) = RF(98) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(18)), (EG(5)*EG(17)));
    RB(99) = RF(99) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(18)), (EG(6)*EG(17)));
    RB(100) = RF(100) * MIN(rtemp_inv, SMALL_INV);

