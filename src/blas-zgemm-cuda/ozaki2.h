#ifndef OZAKI2_H
#define OZAKI2_H

// Ozaki-II emulation of ZGEMM on INT8 matrix engines, shared by the CUDA, HIP
// and SYCL versions of the benchmark.
//
// The complex inputs are scaled row-wise (A) and column-wise (B) to integers
// A', B' with |A'|,|B'| <= 2^bits, so that C' = A'B' is an exact integer
// matrix with |C'| < M/4, M = p_1 * ... * p_N.  C' is computed modulo each
// p_j with INT8 GEMMs and reconstructed with the Chinese Remainder Theorem.
//
// 4M: Cr' = Ar'Br' - Ai'Bi', Ci' = Ar'Bi' + Ai'Br'  (four real products)
// 2M: every modulus has a square root of -1 (iota), so modulo p_j
//       C+ = (Ar' + iota Ai')(Br' + iota Bi'),  C- = (Ar' - iota Ai')(Br' - iota Bi')
//       Cr' = (C+ + C-) / 2,                    Ci' = (C+ - C-) / (2 iota)
//     needs two real products only.  All operations are exact, so 2M and 4M
//     produce bitwise identical results.
//
// Apart from the final reconstruction, the element-wise work uses integer
// arithmetic, since FP64 throughput is low on many GPUs, or floating point
// where that is cheaper (OZ_FP_ELEMENTWISE, see the element-wise API below).

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OZ_HD __host__ __device__
#else
#define OZ_HD
#endif

// Pairwise-coprime moduli < 256 whose prime factors are all 1 (mod 4), so
// that -1 is a quadratic residue (needed by 2M); symmetric residues
// [-(p-1)/2, (p-1)/2] fit in int8.  The product of all 17 is < 2^127.
#define OZ_MAX_MODULI 17
static const int kOzModuli[OZ_MAX_MODULI] = {241, 233, 229, 221, 205, 197,
                                             193, 181, 173, 157, 149, 137,
                                             113, 109, 101, 97,  89};

// Padding of the INT8 operands (zero filled) so that the INT8 GEMMs only see
// tile-aligned shapes.
#define OZ_MN_ALIGN 128
#define OZ_K_ALIGN 64

// Largest |residue| is 120, so an INT32 dot product of length k is exact as
// long as 2 * k * 120^2 < 2^31 (the factor 2 covers the 4M combinations).
#define OZ_MAX_K 74000

// The columns of C index the second grid dimension of the kernels, whose
// size is limited to 65535 work-groups on some devices.
#define OZ_MAX_N 65535

// Fractional bits of the low CRT weights (see ozaki_setup).
#define OZ_LO_FRAC 49

struct OzakiParams {
  int num_moduli;
  int bits;                          // |A'|,|B'| <= 2^bits
  int shift;                         // CRT sums are in units of 2^shift
  int p[OZ_MAX_MODULI];
  unsigned mu[OZ_MAX_MODULI];        // floor(2^32 / p), Barrett reduction
  int iota[OZ_MAX_MODULI];           // iota^2 = -1 (mod p)
  int iota2[OZ_MAX_MODULI];          // 2 iota (mod p)
  double rpd[OZ_MAX_MODULI];         // 1 / p rounded, for floating-point
  float rpf[OZ_MAX_MODULI];          //   reductions (OZ_FP_ELEMENTWISE)
  // Limb weights of the residues (see oz_residues): row 0 = {1, 2^18, 2^36},
  // row 1 = iota * row 0, row 2 = (p - iota) * row 0, all mod p; rofs[r]
  // removes the bias 2^54 from Re (r = 0), Re + iota Im (1), Re - iota Im (2).
  unsigned rw[OZ_MAX_MODULI][3][3];
  unsigned rofs[OZ_MAX_MODULI][3];
  // CRT weights of the residues u_j = 2 Re and v_j = 2 iota Im (mod p_j) of
  // the products (folding the halving into the weights saves reductions):
  //   w_j = (M/p_j)(((M/p_j)^-1 t_j) mod p_j),  t_j = 2^-1 (u), (2 iota)^-1 (v)
  //       = W1_j 2^shift + W2_j 2^(shift - OZ_LO_FRAC)   (W2_j rounded)
  // stored as 19-bit limbs of W1_j (<= 2^40) and W2_j + 2^48 (<= 2^49), so
  // that the CRT sums only need 24-bit multiplies (see OzCrt).
  unsigned cw[OZ_MAX_MODULI][2][6];
  // The same as doubles {W1_j, W2h_j, W2l_j} with W2_j = W2h_j 2^24 + W2l_j,
  // 0 <= W2l_j < 2^24: sums of u_j times them are exact in FP64
  // (OZ_FP_ELEMENTWISE).
  double cwd[OZ_MAX_MODULI][2][3];
  long long M1, M2;                  // the split of M
  float qscalef;                     // 2^shift / M
  double log2M;
};

// CRT partial sums of one complex element across passes over the moduli.
struct alignas(32) OzAcc {
  long long r1, r2, i1, i2;
};

static inline size_t oz_round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

static int oz_mod_inverse(int a, int p) {
  a %= p;
  if (a < 0) a += p;
  for (int x = 1; x < p; x++)
    if ((a * x) % p == 1) return x;
  return 0;
}

static int oz_sqrt_minus_one(int p) {
  for (int t = 1; t < p; t++)
    if ((t * t) % p == p - 1) return t;
  return 0;
}

// Splits v = hi 2^shift + lo 2^(shift - OZ_LO_FRAC), hi rounded to nearest.
static void oz_split(unsigned __int128 v, int shift, long long &hi, long long &lo) {
  unsigned __int128 h = v;
  if (shift > 0) h = (v + ((unsigned __int128)1 << (shift - 1))) >> shift;
  hi = (long long)h;
  __int128 r = (__int128)v - (__int128)(h << shift);  // |r| <= 2^(shift-1)
  if (shift >= OZ_LO_FRAC) {
    const int s = shift - OZ_LO_FRAC;
    lo = (long long)(s > 0 ? (r + ((__int128)1 << (s - 1))) >> s : r);
  } else {
    lo = (long long)(r * ((__int128)1 << (OZ_LO_FRAC - shift)));
  }
}

// Sets up the moduli and CRT constants for `num_moduli` moduli and a
// contraction length k.  Returns false if the configuration is not usable.
static bool ozaki_setup(int num_moduli, int k, OzakiParams &P) {
  if (num_moduli < 1 || num_moduli > OZ_MAX_MODULI) return false;
  P.num_moduli = num_moduli;

  unsigned __int128 M = 1;
  P.log2M = 0.0;
  for (int j = 0; j < num_moduli; j++) {
    const int p = kOzModuli[j];
    M *= (unsigned)p;
    P.log2M += std::log2((double)p);
    P.p[j] = p;
    P.mu[j] = (unsigned)((1ull << 32) / (unsigned)p);
    P.iota[j] = oz_sqrt_minus_one(p);
    P.iota2[j] = 2 * P.iota[j] % p;
    P.rpd[j] = 1.0 / p;
    P.rpf[j] = 1.0f / p;

    const unsigned long long w[3] = {1, (1ull << 18) % p, (1ull << 36) % p};
    const unsigned long long c[3] = {1, (unsigned)P.iota[j], (unsigned)(p - P.iota[j])};
    for (int r = 0; r < 3; r++)
      for (int l = 0; l < 3; l++) P.rw[j][r][l] = (unsigned)(c[r] * w[l] % p);
    const unsigned long long bias = (1ull << 54) % p;
    P.rofs[j][0] = (unsigned)((p - bias) % p);
    P.rofs[j][1] = (unsigned)((p - bias * (1 + c[1]) % p) % p);
    P.rofs[j][2] = (unsigned)((p - bias * (1 + c[2]) % p) % p);
  }

  // 2 k 2^(2 bits) <= M / 4 keeps |C'| well inside (-M/2, M/2).
  int bits = (int)std::floor((P.log2M - 3.0 - std::log2((double)k)) / 2.0);
  if (bits > 53) bits = 53;  // no gain beyond the FP64 significand
  if (bits < 1) return false;
  P.bits = bits;

  // Keep W1 below 2^40 so that sum_j c_j W1_j and the quotient times M1 stay
  // far from int64 overflow and convert exactly to double.
  int W = 0;
  for (unsigned __int128 t = M; t; t >>= 1) W++;
  P.shift = (W > 40) ? W - 40 : 0;

  for (int j = 0; j < num_moduli; j++) {
    const unsigned __int128 Mj = M / (unsigned)P.p[j];
    const int p = P.p[j];
    const int y = oz_mod_inverse((int)(Mj % (unsigned)p), p);
    const int t[2] = {(p + 1) / 2, oz_mod_inverse(P.iota2[j], p)};
    for (int c = 0; c < 2; c++) {
      long long W1, W2;
      oz_split(Mj * (unsigned)(y * t[c] % p), P.shift, W1, W2);  // < M
      const unsigned long long w[2] = {(unsigned long long)W1,
                                       (unsigned long long)(W2 + (1ll << 48))};
      for (int h = 0; h < 2; h++)
        for (int l = 0; l < 3; l++)
          P.cw[j][c][3 * h + l] = (unsigned)(w[h] >> (19 * l)) & 0x7ffffu;
      const long long W2l = W2 & 0xffffff;
      P.cwd[j][c][0] = (double)W1;
      P.cwd[j][c][1] = (double)((W2 - W2l) / (1 << 24));
      P.cwd[j][c][2] = (double)W2l;
    }
  }
  oz_split(M, P.shift, P.M1, P.M2);
  P.qscalef = (float)(std::ldexp(1.0, P.shift) / (double)M);
  return true;
}

// ---- element-wise helpers (host and device) ---------------------------------

// Upper 32 bits of the bit pattern of |v|.  For non-negative doubles the bit
// patterns order like the values, and the upper word holds the whole
// exponent field, so the row (column) maxima are taken on these words, which
// needs no 64-bit atomics.
OZ_HD inline unsigned oz_abs_hi(double v) {
  unsigned long long b;
  memcpy(&b, &v, sizeof(b));
  return (unsigned)(b >> 32) & 0x7fffffffu;
}

// Scale exponent e of a row (column) from oz_abs_hi of its largest
// magnitude: |x| 2^e < 2^bits for all its elements.  Subnormals count as 0.
OZ_HD inline int oz_scale_exponent(unsigned amax_hi, int bits) {
  const int E = (int)(amax_hi >> 20);
  return (E == 0) ? 0 : bits - (E - 1022);
}

// round(v 2^e) as an integer, |v 2^e| <= 2^53 (ties away from zero).
OZ_HD inline long long oz_to_fixed(double v, int e) {
  unsigned long long b;
  memcpy(&b, &v, sizeof(b));
  const int E = (int)((b >> 52) & 0x7ff);
  if (E == 0) return 0;
  const unsigned long long mant = (b & 0xfffffffffffffull) | (1ull << 52);
  const int s = E - 1075 + e;
  unsigned long long mag;
  if (s >= 0) mag = mant << s;
  else if (s > -64) mag = (mant + (1ull << (-s - 1))) >> (-s);
  else mag = 0;
  return (b >> 63) ? -(long long)mag : (long long)mag;
}

OZ_HD inline unsigned oz_umulhi(unsigned a, unsigned b) {
  return (unsigned)(((unsigned long long)a * b) >> 32);
}

// x mod p for any 32-bit unsigned x.
OZ_HD inline unsigned oz_barrett(unsigned x, unsigned p, unsigned mu) {
  const unsigned r = x - oz_umulhi(x, mu) * p;
  return (r >= p) ? r - p : r;
}

// A scaled input |x| <= 2^53 as 18-bit limbs of x + 2^54 (>= 0, < 2^55).
struct OzLimbs {
  unsigned l0, l1, l2;
};

OZ_HD inline OzLimbs oz_limbs(long long x) {
  const unsigned long long u = (unsigned long long)x + (1ull << 54);
  return {(unsigned)u & 0x3ffffu, (unsigned)(u >> 18) & 0x3ffffu,
          (unsigned)(u >> 36) & 0x7ffffu};
}

// Symmetric residue in [-(p-1)/2, (p-1)/2] of t < 2^32.
OZ_HD inline int oz_symmetric(unsigned t, unsigned p, unsigned mu) {
  const int s = (int)oz_barrett(t, p, mu);
  return (s > (int)(p >> 1)) ? s - (int)p : s;
}

// Symmetric residues modulo p_j of a complex input with limbs (re, im):
//   4M: s0 = Re, s1 = Im;  2M: s0 = Re + iota Im, s1 = Re - iota Im.
// The weights are below 2^8 (masked so that the products are 24-bit ones)
// and every sum stays below 2^30.
template <bool IS2M>
OZ_HD inline void oz_residues(const OzLimbs &re, const OzLimbs &im, int j,
                              const OzakiParams &P, int &s0, int &s1) {
  const unsigned p = P.p[j], mu = P.mu[j];
  const unsigned w1 = P.rw[j][0][1] & 0xff, w2 = P.rw[j][0][2] & 0xff;
  const unsigned r = re.l0 + re.l1 * w1 + re.l2 * w2;
  if (IS2M) {
    const unsigned a0 = P.rw[j][1][0] & 0xff, a1 = P.rw[j][1][1] & 0xff, a2 = P.rw[j][1][2] & 0xff;
    const unsigned b0 = P.rw[j][2][0] & 0xff, b1 = P.rw[j][2][1] & 0xff, b2 = P.rw[j][2][2] & 0xff;
    s0 = oz_symmetric(r + im.l0 * a0 + im.l1 * a1 + im.l2 * a2 + (P.rofs[j][1] & 0xff), p, mu);
    s1 = oz_symmetric(r + im.l0 * b0 + im.l1 * b1 + im.l2 * b2 + (P.rofs[j][2] & 0xff), p, mu);
  } else {
    const unsigned o = P.rofs[j][0] & 0xff;
    s0 = oz_symmetric(r + o, p, mu);
    s1 = oz_symmetric(im.l0 + im.l1 * w1 + im.l2 * w2 + o, p, mu);
  }
}

// Four int8 values in one word (little endian).
OZ_HD inline unsigned oz_pack4(const int *s) {
  return (unsigned)(s[0] & 0xff) | ((unsigned)(s[1] & 0xff) << 8) |
         ((unsigned)(s[2] & 0xff) << 16) | ((unsigned)(s[3] & 0xff) << 24);
}

// Residue in [0, p) of an int32.
OZ_HD inline int oz_mod(int x, int j, const OzakiParams &P) {
  const unsigned p = P.p[j];
  const unsigned a = (x < 0) ? 0u - (unsigned)x : (unsigned)x;
  const unsigned r = oz_barrett(a, p, P.mu[j]);
  return (x < 0 && r) ? (int)(p - r) : (int)r;
}

// u = 2 Re and v = 2 iota Im (mod p_j) of an element of C' from the 2M
// products C+- = Re +- iota Im, or from the four 4M products.  The int32 sums
// cannot overflow (see OZ_MAX_K).
OZ_HD inline void oz_uv_2m(int cp, int cm, int j, const OzakiParams &P,
                           unsigned &u, unsigned &v) {
  u = (unsigned)oz_mod(cp + cm, j, P);
  v = (unsigned)oz_mod(cp - cm, j, P);
}

OZ_HD inline void oz_uv_4m(int rr, int ir, int ri, int ii, int j,
                           const OzakiParams &P, unsigned &u, unsigned &v) {
  const unsigned p = P.p[j], a = 2u * (unsigned)oz_mod(rr - ii, j, P);
  u = (a >= p) ? a - p : a;
  v = (unsigned)oz_mod(oz_mod(ri + ir, j, P) * P.iota2[j], j, P);
}

// CRT sums of one complex element over the moduli of a pass, as columns of
// 19-bit limb products (each column stays below 17 * 255 * 2^19 < 2^32) and
// the sums of u and v that remove the bias of the W2 limbs.
struct OzCrt {
  unsigned r[6], i[6], nr, ni;
};

OZ_HD inline void oz_crt_add(OzCrt &a, unsigned u, unsigned v, int j,
                             const OzakiParams &P) {
  u &= 0xffu;
  v &= 0xffu;
  for (int l = 0; l < 6; l++) {
    a.r[l] += u * (P.cw[j][0][l] & 0x7ffffu);
    a.i[l] += v * (P.cw[j][1][l] & 0x7ffffu);
  }
  a.nr += u;
  a.ni += v;
}

OZ_HD inline long long oz_limb_sum(const unsigned *c) {
  return (long long)c[0] + ((long long)c[1] << 19) + ((long long)c[2] << 38);
}

// Adds the sums of a pass to s1 = sum u_j W1_j, s2 = sum u_j W2_j (and the
// same for v).
OZ_HD inline void oz_crt_fold(const OzCrt &a, OzAcc &s) {
  s.r1 += oz_limb_sum(a.r);
  s.r2 += oz_limb_sum(a.r + 3) - ((long long)a.nr << 48);
  s.i1 += oz_limb_sum(a.i);
  s.i2 += oz_limb_sum(a.i + 3) - ((long long)a.ni << 48);
}

// ---- element-wise API of the kernels ------------------------------------------
//
// The residues and the CRT sums use integer arithmetic, or, with
// OZ_FP_ELEMENTWISE, floating point, which is cheaper on AMD GPUs (CDNA:
// FP64 at the full or half FP32 rate, quarter-rate 32-bit integer
// multiplies) and the default there.  Both give the same results: the
// symmetric residues are unique and the CRT sums are exact.
//   OzFixed x = oz_fixed(re, im, e);               a scaled complex input
//   oz_residue_pair<IS2M>(x, j, P, s0, s1);         its residues modulo p_j
//   OzSums a = {};  oz_add_2m(a, cp, cm, j, P);     CRT sums of the products
//                   oz_add_4m(a, rr, ir, ri, ii, j, P);
//   oz_fold(a, s);                                  added to the OzAcc s
#if defined(__AMDGCN__) && !defined(OZ_INT_ELEMENTWISE) && !defined(OZ_FP_ELEMENTWISE)
#define OZ_FP_ELEMENTWISE
#endif

#ifdef OZ_FP_ELEMENTWISE

// round(v 2^e) (ties away from zero, subnormals as 0 like oz_to_fixed).
OZ_HD inline double oz_dfixed(double v, int e) {
  return (fabs(v) < DBL_MIN) ? 0.0 : round(ldexp(v, e));
}

// x - q p for an integer |x| <= 2^53 with q = rint(x / p) up to an error of
// 0.0225 (the rounding of 1/p and of the product), so |x - q p| < 0.53 p.
OZ_HD inline float oz_dreduce(double x, double p, double rp) {
  return (float)fma(-rint(x * rp), p, x);
}

// Symmetric residue of an integer |t| < 2^15: the error of t / p is below
// 2^-8 / p, while t / p is at least 1 / (2p) away from a tie.
OZ_HD inline int oz_fsymmetric(float t, float p, float rp) {
  return (int)fmaf(-rintf(t * rp), p, t);
}

// x mod p in [0, p) for any int32 x: floor(x / p + 2^-20) is floor(x / p),
// as the rounding errors (< 2^-26) and 2^-20 stay below the gaps 1/p of x/p.
OZ_HD inline double oz_dmod(int x, double p, double rp) {
  const double d = x;
  return fma(-floor(fma(d, rp, 0x1p-20)), p, d);
}

struct OzFixed {
  double re, im;
};

OZ_HD inline OzFixed oz_fixed(double re, double im, int e) {
  return {oz_dfixed(re, e), oz_dfixed(im, e)};
}

template <bool IS2M>
OZ_HD inline void oz_residue_pair(const OzFixed &x, int j, const OzakiParams &P,
                                  int &s0, int &s1) {
  const double pd = P.p[j], rpd = P.rpd[j];
  const float pf = P.p[j], rpf = P.rpf[j], io = P.iota[j];
  const float xr = oz_dreduce(x.re, pd, rpd), xi = oz_dreduce(x.im, pd, rpd);
  if (IS2M) {
    s0 = oz_fsymmetric(fmaf(io, xi, xr), pf, rpf);
    s1 = oz_fsymmetric(fmaf(-io, xi, xr), pf, rpf);
  } else {
    s0 = oz_fsymmetric(xr, pf, rpf);
    s1 = oz_fsymmetric(xi, pf, rpf);
  }
}

// Sums of u_j and v_j times the weights cwd: every partial sum is an integer
// below 2^53, so they are exact.
struct OzSums {
  double r[3], i[3];
};

OZ_HD inline void oz_add_uv(OzSums &a, double u, double v, int j, const OzakiParams &P) {
  for (int l = 0; l < 3; l++) {
    a.r[l] = fma(u, P.cwd[j][0][l], a.r[l]);
    a.i[l] = fma(v, P.cwd[j][1][l], a.i[l]);
  }
}

OZ_HD inline void oz_add_2m(OzSums &a, int cp, int cm, int j, const OzakiParams &P) {
  const double pd = P.p[j], rpd = P.rpd[j];
  oz_add_uv(a, oz_dmod(cp + cm, pd, rpd), oz_dmod(cp - cm, pd, rpd), j, P);
}

OZ_HD inline void oz_add_4m(OzSums &a, int rr, int ir, int ri, int ii, int j,
                            const OzakiParams &P) {
  unsigned u, v;
  oz_uv_4m(rr, ir, ri, ii, j, P, u, v);
  oz_add_uv(a, u, v, j, P);
}

OZ_HD inline void oz_fold(const OzSums &a, OzAcc &s) {
  s.r1 += (long long)a.r[0];
  s.r2 += (long long)a.r[1] * (1 << 24) + (long long)a.r[2];
  s.i1 += (long long)a.i[0];
  s.i2 += (long long)a.i[1] * (1 << 24) + (long long)a.i[2];
}

#else  // integer arithmetic

struct OzFixed {
  OzLimbs re, im;
};

OZ_HD inline OzFixed oz_fixed(double re, double im, int e) {
  return {oz_limbs(oz_to_fixed(re, e)), oz_limbs(oz_to_fixed(im, e))};
}

template <bool IS2M>
OZ_HD inline void oz_residue_pair(const OzFixed &x, int j, const OzakiParams &P,
                                  int &s0, int &s1) {
  oz_residues<IS2M>(x.re, x.im, j, P, s0, s1);
}

typedef OzCrt OzSums;

OZ_HD inline void oz_add_2m(OzSums &a, int cp, int cm, int j, const OzakiParams &P) {
  unsigned u, v;
  oz_uv_2m(cp, cm, j, P, u, v);
  oz_crt_add(a, u, v, j, P);
}

OZ_HD inline void oz_add_4m(OzSums &a, int rr, int ir, int ri, int ii, int j,
                            const OzakiParams &P) {
  unsigned u, v;
  oz_uv_4m(rr, ir, ri, ii, j, P, u, v);
  oz_crt_add(a, u, v, j, P);
}

OZ_HD inline void oz_fold(const OzSums &a, OzAcc &s) { oz_crt_fold(a, s); }

#endif

// C' / 2^shift from the CRT sums s1 = sum c_j W1_j and s2 = sum c_j W2_j.
// The quotient q = round(s1 2^shift / M) is below 2^13 and s1 2^shift / M is
// within 1/4 + 2^-28 of it (|C'| < M/4), so FP32 rounds to the same q.
OZ_HD inline double oz_reconstruct(long long s1, long long s2, const OzakiParams &P) {
  const long long q = (long long)rintf((float)s1 * P.qscalef);
  const long long t = s1 - q * P.M1;  // exact, |t| < 2^41
  const long long u = s2 - q * P.M2;
  return (double)t + ldexp((double)u, -OZ_LO_FRAC);
}

// ---- host utilities -----------------------------------------------------------

// Number of moduli processed per pass so that the INT8 operands and INT32
// products of one pass fit into `budget` bytes.
static int ozaki_group_size(int num_moduli, bool is2m, size_t mp, size_t np,
                            size_t kp, size_t budget) {
  const size_t per = 2 * (mp + np) * kp + (is2m ? 2 : 4) * mp * np * sizeof(int32_t);
  size_t g = budget / per;
  if (g < 1) g = 1;
  if (g > (size_t)num_moduli) g = num_moduli;
  return (int)g;
}

// Naive host ZGEMM on interleaved complex data (column-major, C = A * B).
static void reference_zgemm(int m, int n, int k, const double *A,
                            const double *B, double *C) {
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m; i++) {
      double re = 0.0, im = 0.0;
      for (int p = 0; p < k; p++) {
        const double ar = A[2 * ((size_t)p * m + i)], ai = A[2 * ((size_t)p * m + i) + 1];
        const double br = B[2 * ((size_t)j * k + p)], bi = B[2 * ((size_t)j * k + p) + 1];
        re += ar * br - ai * bi;
        im += ar * bi + ai * br;
      }
      C[2 * ((size_t)j * m + i)] = re;
      C[2 * ((size_t)j * m + i) + 1] = im;
    }
}

// Error of `test` relative to `ref` (interleaved complex data):
//   normwise      = max |test - ref| / max |ref|
//   componentwise = max |test - ref| / |ref|
static void compare(const double *test, const double *ref, size_t count,
                    double &normwise, double &componentwise) {
  double max_diff = 0.0, max_ref = 0.0;
  componentwise = 0.0;
  for (size_t i = 0; i < count; i++) {
    const double d = std::hypot(test[2 * i] - ref[2 * i], test[2 * i + 1] - ref[2 * i + 1]);
    const double r = std::hypot(ref[2 * i], ref[2 * i + 1]);
    if (d > max_diff) max_diff = d;
    if (r > max_ref) max_ref = r;
    const double c = d / (r + 1e-300);
    if (c > componentwise) componentwise = c;
  }
  normwise = max_diff / (max_ref + 1e-300);
}

#endif
