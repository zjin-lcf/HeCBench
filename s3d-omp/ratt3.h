    const real TEMP = T[i]*tconv;
    //real ALOGT = LOG(TEMP);
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(3)*EG(8)), (EG(5)*EG(7)));
    RB(26) = RF(26) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(27) = RF(27) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(28) = RF(28) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(14)*PFAC), EG(15));
    RB(29) = RF(29) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(14)), (EG(2)*EG(15)));
    RB(30) = RF(30) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(14)*PFAC), EG(17));
    RB(31) = RF(31) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(14)), (EG(3)*EG(15)));
    RB(32) = RF(32) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(14)), (EG(5)*EG(15)));
    RB(33) = RF(33) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(9)), (EG(2)*EG(14)));
    RB(34) = RF(34) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(9)), (EG(2)*EG(16)));
    RB(35) = RF(35) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(9)), (EG(2)*EG(10)));
    RB(36) = RF(36) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(6)*EG(9)), (EG(2)*EG(17)));
    RB(37) = RF(37) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(9)), (EG(3)*EG(16)));
    RB(38) = RF(38) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(14)*PFAC), EG(25));
    RB(39) = RF(39) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(15)), (EG(14)*EG(16)));
    RB(40) = RF(40) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)*PFAC), EG(17));
    RB(41) = RF(41) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)), (EG(1)*EG(14)));
    RB(42) = RF(42) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(5)*EG(14)));
    RB(43) = RF(43) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(2)*EG(15)));
    RB(44) = RF(44) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(16)), (EG(6)*EG(14)));
    RB(45) = RF(45) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*PFAC), (EG(2)*EG(14)));
    RB(46) = RF(46) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(16)), (EG(7)*EG(14)));
    RB(47) = RF(47) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(10)*PFAC), EG(12));
    RB(48) = RF(48) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(10)), (EG(2)*EG(12)));
    RB(49) = RF(49) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(10)), (EG(2)*EG(16)));
    RB(50) = RF(50) * MIN(rtemp_inv, SMALL_INV);

