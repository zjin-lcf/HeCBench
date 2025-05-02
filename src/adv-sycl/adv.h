// begin of advCubatureHex3D
void advCubatureHex3D (
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const int Nelements,
    const dfloat * __restrict vgeo,
    const dfloat * __restrict cubvgeo,
    const dfloat * __restrict cubDiffInterpT,
    const dfloat * __restrict cubInterpT,
    const int offset,
    const dfloat * __restrict u,
          dfloat * __restrict adv)
{
  auto cgf = [&](sycl::handler& cgh) {
    sycl::local_accessor<dfloat, 2> s_cubD(sycl::range<2>{16,16}, cgh);
    sycl::local_accessor<dfloat, 2> s_cubInterpT(sycl::range<2>{8,16}, cgh);
    sycl::local_accessor<dfloat, 2> s_U(sycl::range<2>{8,8}, cgh);
    sycl::local_accessor<dfloat, 2> s_V(sycl::range<2>{8,8}, cgh);
    sycl::local_accessor<dfloat, 2> s_W(sycl::range<2>{8,8}, cgh);
    sycl::local_accessor<dfloat, 2> s_U1(sycl::range<2>{16,16}, cgh);
    sycl::local_accessor<dfloat, 2> s_V1(sycl::range<2>{16,16}, cgh);
    sycl::local_accessor<dfloat, 2> s_W1(sycl::range<2>{16,16}, cgh);

    auto kfn = [=](sycl::nd_item<3> item) {
      dfloat r_U[16], r_V[16], r_W[16];
      dfloat r_Ud[16], r_Vd[16], r_Wd[16];

      const int e = item.get_group(2);
      const int i = item.get_local_id(2);
      const int j = item.get_local_id(1);
      const int id = j * 16 + i;

      if (id < 8 * 16) s_cubInterpT[j][i] = cubInterpT[id];
      s_cubD[j][i] = cubDiffInterpT[id];

      for (int k = 0; k < 16; ++k) {
        r_U[k] = 0;
        r_V[k] = 0;
        r_W[k] = 0;
        r_Ud[k] = 0;
        r_Vd[k] = 0;
        r_Wd[k] = 0;
      }

      for (int c = 0; c < 8; ++c) {
        if (j < 8 && i < 8) {
          const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
          s_U[j][i] = u[id + 0 * offset];
          s_V[j][i] = u[id + 1 * offset];
          s_W[j][i] = u[id + 2 * offset];
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (j < 8) {
          dfloat U1 = 0, V1 = 0, W1 = 0;
          for (int a = 0; a < 8; ++a) {
            dfloat Iia = s_cubInterpT[a][i];
            U1 += Iia * s_U[j][a];
            V1 += Iia * s_V[j][a];
            W1 += Iia * s_W[j][a];
          }
          s_U1[j][i] = U1;
          s_V1[j][i] = V1;
          s_W1[j][i] = W1;
        } else {
          s_U1[j][i] = 0;
          s_V1[j][i] = 0;
          s_W1[j][i] = 0;
        }

        item.barrier(sycl::access::fence_space::local_space);

        dfloat U2 = 0, V2 = 0, W2 = 0;
        for (int b = 0; b < 8; ++b) {
          dfloat Ijb = s_cubInterpT[b][j];
          U2 += Ijb * s_U1[b][i];
          V2 += Ijb * s_V1[b][i];
          W2 += Ijb * s_W1[b][i];
        }
        for (int k = 0; k < 16; ++k) {
          dfloat Ikc = s_cubInterpT[c][k];
          r_U[k] += Ikc * U2;
          r_V[k] += Ikc * V2;
          r_W[k] += Ikc * W2;
        }
        for (int k = 0; k < 16; ++k) {
          r_Ud[k] = r_U[k];
          r_Vd[k] = r_V[k];
          r_Wd[k] = r_W[k];
        }
      }

      for (int k = 0; k < 16; ++k) {
        s_U1[j][i] = r_Ud[k];
        s_V1[j][i] = r_Vd[k];
        s_W1[j][i] = r_Wd[k];

        item.barrier(sycl::access::fence_space::local_space);

        dfloat Udr = 0, Uds = 0, Udt = 0;
        dfloat Vdr = 0, Vds = 0, Vdt = 0;
        dfloat Wdr = 0, Wds = 0, Wdt = 0;
        for (int n = 0; n < 16; ++n) {
          dfloat Din = s_cubD[i][n];
          Udr += Din * s_U1[j][n];
          Vdr += Din * s_V1[j][n];
          Wdr += Din * s_W1[j][n];
        }
        for (int n = 0; n < 16; ++n) {
          dfloat Djn = s_cubD[j][n];
          Uds += Djn * s_U1[n][i];
          Vds += Djn * s_V1[n][i];
          Wds += Djn * s_W1[n][i];
        }
        for (int n = 0; n < 16; ++n) {
          dfloat Dkn = s_cubD[k][n];
          Udt += Dkn * r_Ud[n];
          Vdt += Dkn * r_Vd[n];
          Wdt += Dkn * r_Wd[n];
        }

        const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
        const dfloat drdx = cubvgeo[gid + p_RXID * p_cubNp];
        const dfloat drdy = cubvgeo[gid + p_RYID * p_cubNp];
        const dfloat drdz = cubvgeo[gid + p_RZID * p_cubNp];
        const dfloat dsdx = cubvgeo[gid + p_SXID * p_cubNp];
        const dfloat dsdy = cubvgeo[gid + p_SYID * p_cubNp];
        const dfloat dsdz = cubvgeo[gid + p_SZID * p_cubNp];
        const dfloat dtdx = cubvgeo[gid + p_TXID * p_cubNp];
        const dfloat dtdy = cubvgeo[gid + p_TYID * p_cubNp];
        const dfloat dtdz = cubvgeo[gid + p_TZID * p_cubNp];
        const dfloat JW = cubvgeo[gid + p_JWID * p_cubNp];
        const dfloat Un = r_U[k];
        const dfloat Vn = r_V[k];
        const dfloat Wn = r_W[k];
        const dfloat Uhat = JW * (Un * drdx + Vn * drdy + Wn * drdz);
        const dfloat Vhat = JW * (Un * dsdx + Vn * dsdy + Wn * dsdz);
        const dfloat What = JW * (Un * dtdx + Vn * dtdy + Wn * dtdz);
        r_U[k] = Uhat * Udr + Vhat * Uds + What * Udt;
        r_V[k] = Uhat * Vdr + Vhat * Vds + What * Vdt;
        r_W[k] = Uhat * Wdr + Vhat * Wds + What * Wdt;
      }

      for (int c = 0; c < 8; ++c) {
        dfloat rhsU = 0, rhsV = 0, rhsW = 0;
        for (int k = 0; k < 16; ++k) {
          dfloat Ikc = s_cubInterpT[c][k];
          rhsU += Ikc * r_U[k];
          rhsV += Ikc * r_V[k];
          rhsW += Ikc * r_W[k];
        }

        if (i < 8 && j < 8) {
          s_U[j][i] = rhsU;
          s_V[j][i] = rhsV;
          s_W[j][i] = rhsW;
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (j < 8) {
          dfloat rhsU = 0, rhsV = 0, rhsW = 0;
          for (int k = 0; k < 16; ++k) {
            dfloat Ijb = s_cubInterpT[j][k];
            if (k < 8 && i < 8) {
              rhsU += Ijb * s_U[k][i];
              rhsV += Ijb * s_V[k][i];
              rhsW += Ijb * s_W[k][i];
            }
          }
          s_U1[j][i] = rhsU;
          s_V1[j][i] = rhsV;
          s_W1[j][i] = rhsW;
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (i < 8 && j < 8) {
          dfloat rhsU = 0, rhsV = 0, rhsW = 0;
          for (int k = 0; k < 16; ++k) {
            dfloat Iia = s_cubInterpT[i][k];
            rhsU += Iia * s_U1[j][k];
            rhsV += Iia * s_V1[j][k];
            rhsW += Iia * s_W1[j][k];
          }
          const int gid = e * p_Np * p_Nvgeo + c * 8 * 8 + j * 8 + i;
          const dfloat IJW = vgeo[gid + p_IJWID * p_Np];
          const int id = e * p_Np + c * 8 * 8 + j * 8 + i;
          adv[id + 0 * offset] = IJW * rhsU;
          adv[id + 1 * offset] = IJW * rhsV;
          adv[id + 2 * offset] = IJW * rhsW;
        }
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}
// end of advCubatureHex3D
