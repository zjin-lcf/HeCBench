
    const real TEMP = T[i]*tconv;
    //real ALOGT = LOG(TEMP);

    const real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(4)), (EG(3)*EG(5)));
    RB(1) = RF(1) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(3)), (EG(2)*EG(5)));
    RB(2) = RF(2) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(5)), (EG(2)*EG(6)));
    RB(3) = RF(3) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)), (EG(3)*EG(6)));
    RB(4) = RF(4) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(5) = RF(5) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(6) = RF(6) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(7) = RF(7) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(8) = RF(8) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(5)*PFAC), EG(6));
    RB(9) = RF(9) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(3)*PFAC), EG(5));
    RB(10) = RF(10) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(3)*PFAC), EG(4));
    RB(11) = RF(11) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(12) = RF(12) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(13) = RF(13) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(14) = RF(14) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(15) = RF(15) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)*PFAC), EG(8));
    RB(16) = RF(16) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(3)*EG(6)));
    RB(17) = RF(17) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(1)*EG(4)));
    RB(18) = RF(18) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(5)*EG(5)));
    RB(19) = RF(19) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(7)), (EG(4)*EG(5)));
    RB(20) = RF(20) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(7)), (EG(4)*EG(6)));
    RB(21) = RF(21) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(22) = RF(22) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(23) = RF(23) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(1)*EG(7)));
    RB(24) = RF(24) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(5)*EG(6)));
    RB(25) = RF(25) * MIN(rtemp_inv, SMALL_INV);

