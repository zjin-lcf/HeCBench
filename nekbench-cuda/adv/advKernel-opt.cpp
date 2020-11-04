
__global__ void advCubatureHex3D (
    const int Nelements,
    const double * __restrict__ vgeo,
    const double * __restrict__ cubvgeo,
    const double * __restrict__ cubD,
    const double * __restrict__ cubInterpT,
    const int offset,
    const double * __restrict__ U,
    double * __restrict__ NU) {
  {
    int e = blockIdx.x;
    __shared__ double s_cubD[256];
    __shared__ double s_cubInterpT[128];
    __shared__ double s_U[64];
    __shared__ double s_V[64];
    __shared__ double s_W[64];
    __shared__ double s_U1[256];
    __shared__ double s_V1[256];
    __shared__ double s_W1[256];
    double r_U[16], r_V[16], r_W[16];
    double r_Ud[16], r_Vd[16], r_Wd[16];
    {
      int j = threadIdx.y;
      {
        int i = threadIdx.x;
        const int id = i + j * 16;
        if (id < 128) {
          s_cubInterpT[id] = cubInterpT[id];
        }
        s_cubD[id] = cubD[id];
        for (int k = 0; k < 16; ++k) {
          r_U[k] = 0;
          r_V[k] = 0;
          r_W[k] = 0;
          r_Ud[k] = 0;
          r_Vd[k] = 0;
          r_Wd[k] = 0;
        }
      }
    }
    for (int c = 0; c < 8; ++c) {
       __syncthreads();
      {
        int b = 0 + threadIdx.y;
        {
          int a = 0 + threadIdx.x;
          if (a < 8 && b < 8) {
            const int id = e * p_Np + c * 8 * 8 + b * 8 + a;
            s_U[b*8+a] = U[id + 0 * offset];
            s_V[b*8+a] = U[id + 1 * offset];
            s_W[b*8+a] = U[id + 2 * offset];
          }
        }
      }
       __syncthreads();
      {
        int b = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          if (b < 8) {
            double U1 = 0, V1 = 0, W1 = 0;
            for (int a = 0; a < 8; ++a) {
              double Iia = s_cubInterpT[a*16+i];
              U1 += Iia * s_U[b*8+a];
              V1 += Iia * s_V[b*8+a];
              W1 += Iia * s_W[b*8+a];
            }
            s_U1[b*16+i] = U1;
            s_V1[b*16+i] = V1;
            s_W1[b*16+i] = W1;
          }
        }
      }
       __syncthreads();
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          double U2 = 0, V2 = 0, W2 = 0;
          for (int b = 0; b < 8; ++b) {
            double Ijb = s_cubInterpT[b][j];
            U2 += Ijb * s_U1[b][i];
            V2 += Ijb * s_V1[b][i];
            W2 += Ijb * s_W1[b][i];
          }
          for (int k = 0; k < 16; ++k) {
            double Ikc = s_cubInterpT[c][k];
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
      }
    }
    for (int k = 0; k < 16; ++k) {
       __syncthreads();
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          s_U1[j][i] = r_Ud[k];
          s_V1[j][i] = r_Vd[k];
          s_W1[j][i] = r_Wd[k];
        }
      }
       __syncthreads();
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          double Udr = 0, Uds = 0, Udt = 0;
          double Vdr = 0, Vds = 0, Vdt = 0;
          double Wdr = 0, Wds = 0, Wdt = 0;
          for (int n = 0; n < 16; ++n) {
            double Din = s_cubD[i*16+n];
            Udr += Din * s_U1[j*16+n];
            Vdr += Din * s_V1[j*16+n];
            Wdr += Din * s_W1[j*16+n];
          }
          for (int n = 0; n < 16; ++n) {
            double Djn = s_cubD[j*16+n];
            Uds += Djn * s_U1[n*16+i];
            Vds += Djn * s_V1[n*16+i];
            Wds += Djn * s_W1[n*16+i];
          }
          for (int n = 0; n < 16; ++n) {
            double Dkn = s_cubD[k*16+n];
            Udt += Dkn * r_Ud[n];
            Vdt += Dkn * r_Vd[n];
            Wdt += Dkn * r_Wd[n];
          }
          const int gid = e * p_cubNp * p_Nvgeo + k * 16 * 16 + j * 16 + i;
          const double drdx = cubvgeo[gid + p_RXID * p_cubNp];
          const double drdy = cubvgeo[gid + p_RYID * p_cubNp];
          const double drdz = cubvgeo[gid + p_RZID * p_cubNp];
          const double dsdx = cubvgeo[gid + p_SXID * p_cubNp];
          const double dsdy = cubvgeo[gid + p_SYID * p_cubNp];
          const double dsdz = cubvgeo[gid + p_SZID * p_cubNp];
          const double dtdx = cubvgeo[gid + p_TXID * p_cubNp];
          const double dtdy = cubvgeo[gid + p_TYID * p_cubNp];
          const double dtdz = cubvgeo[gid + p_TZID * p_cubNp];
          const double JW = cubvgeo[gid + p_JWID * p_cubNp];
          const double Un = r_U[k];
          const double Vn = r_V[k];
          const double Wn = r_W[k];
          const double Uhat = JW * (Un * drdx + Vn * drdy + Wn * drdz);
          const double Vhat = JW * (Un * dsdx + Vn * dsdy + Wn * dsdz);
          const double What = JW * (Un * dtdx + Vn * dtdy + Wn * dtdz);
          r_U[k] = Uhat * Udr + Vhat * Uds + What * Udt;
          r_V[k] = Uhat * Vdr + Vhat * Vds + What * Vdt;
          r_W[k] = Uhat * Wdr + Vhat * Wds + What * Wdt;
        }
      }
    }
    for (int c = 0; c < 8; ++c) {
       __syncthreads();
      {
        int j = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          double rhsU = 0, rhsV = 0, rhsW = 0;
          for (int k = 0; k < 16; ++k) {
            double Ikc = s_cubInterpT[c*16+k];
            rhsU += Ikc * r_U[k];
            rhsV += Ikc * r_V[k];
            rhsW += Ikc * r_W[k];
          }
          s_U[j*8+i] = rhsU;
          s_V[j*8+i] = rhsV;
          s_W[j*8+i] = rhsW;
        }
      }
       __syncthreads();
      {
        int b = 0 + threadIdx.y;
        {
          int i = 0 + threadIdx.x;
          if (b < 8) {
            double rhsU = 0, rhsV = 0, rhsW = 0;
            for (int j = 0; j < 16; ++j) {
              double Ijb = s_cubInterpT[b][j];
              rhsU += Ijb * s_U[j*8+i];
              rhsV += Ijb * s_V[j*8+i];
              rhsW += Ijb * s_W[j*8+i];
            }
            s_U1[b*16+i] = rhsU;
            s_V1[b*16+i] = rhsV;
            s_W1[b*16+i] = rhsW;
          }
        }
      }
       __syncthreads();
      {
        int b = 0 + threadIdx.y;
        {
          int a = 0 + threadIdx.x;
          if (a < 8 && b < 8) {
            double rhsU = 0, rhsV = 0, rhsW = 0;
            for (int i = 0; i < 16; ++i) {
              double Iia = s_cubInterpT[a][i];
              rhsU += Iia * s_U1[b*16+i];
              rhsV += Iia * s_V1[b*16+i];
              rhsW += Iia * s_W1[b*16+i];
            }
            const int gid = e * p_Np * p_Nvgeo + c * 8 * 8 + b * 8 + a;
            const double IJW = vgeo[gid + p_IJWID * p_Np];
            const int id = e * p_Np + c * 8 * 8 + b * 8 + a;
            NU[id + 0 * offset] = IJW * rhsU;
            NU[id + 1 * offset] = IJW * rhsV;
            NU[id + 2 * offset] = IJW * rhsW;
          }
        }
      }
    }
  }
}
