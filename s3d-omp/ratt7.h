    const real TEMP = T[i]*tconv;
    //const real ALOGT = LOG(TEMP);
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(26)*PFAC), EG(27));
    RB(126) = RF(126) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(1)*EG(25)));
    RB(127) = RF(127) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(12)*EG(14)));
    RB(128) = RF(128) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(5)*EG(25)));
    RB(129) = RF(129) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(10)*EG(15)));
    RB(130) = RF(130) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(26)), (EG(6)*EG(25)));
    RB(131) = RF(131) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)*PFAC), EG(22));
    RB(132) = RF(132) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(19)));
    RB(133) = RF(133) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(20)));
    RB(134) = RF(134) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(2)*EG(26)));
    RB(135) = RF(135) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(12)*EG(14)));
    RB(136) = RF(136) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(21)), (EG(6)*EG(19)));
    RB(137) = RF(137) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(7)*EG(19)));
    RB(138) = RF(138) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(3)*EG(27)));
    RB(139) = RF(139) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(16)*EG(17)));
    RB(140) = RF(140) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(21)), (EG(5)*EG(27)));
    RB(141) = RF(141) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(21)), (EG(7)*EG(22)));
    RB(142) = RF(142) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(21)), (EG(14)*EG(22)));
    RB(143) = RF(143) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(13)*EG(19)));
    RB(144) = RF(144) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)*PFAC), EG(30));
    RB(145) = RF(145) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(2)*EG(29)));
    RB(146) = RF(146) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(27), (EG(12)*EG(14)*PFAC));
    RB(147) = RF(147) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)*PFAC), EG(28));
    RB(148) = RF(148) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(12)*EG(16)));
    RB(149) = RF(149) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(1)*EG(26)));
    RB(150) = RF(150) * MIN(rtemp_inv, SMALL_INV);

