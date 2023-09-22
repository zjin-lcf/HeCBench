// *****************************************************************************
//                   Copyright (C) 2014, UChicago Argonne, LLC
//                              All Rights Reserved
// 	       High-Performance CRC64 Library (ANL-SF-14-095)
//                    Hal Finkel, Argonne National Laboratory
// 
//                              OPEN SOURCE LICENSE
// 
// Under the terms of Contract No. DE-AC02-06CH11357 with UChicago Argonne, LLC,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the names of UChicago Argonne, LLC or the Department of Energy nor
//    the names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission. 
//  
// *****************************************************************************
//                                  DISCLAIMER
// 
// THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.
// 
// NEITHER THE UNTED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF
// ENERGY, NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY
// WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY
// FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA,
// APPARATUS, PRODUCT, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
// INFRINGE PRIVATELY OWNED RIGHTS.
// 
// *****************************************************************************

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdbool.h>

#include <sycl/sycl.hpp>
#include "CRC64.h"

// The polynomial here is the bit-reversed encoding of 0x42f0e1eba9ea3693.
static const uint64_t crc64_poly = UINT64_C(0xc96c5795d7870f42);

#include "crc64_table.h"

uint64_t crc64_slow(const void *input, size_t nbytes) {
  const unsigned char *data = (const unsigned char*) input;
  uint64_t cs = UINT64_C(0xffffffffffffffff);

  while (nbytes--) {
    uint32_t idx = ((uint32_t) (cs ^ *data++)) & 0xff;
    cs = crc64_table[3][idx] ^ (cs >> 8);
  }

  return cs ^ UINT64_C(0xffffffffffffffff);
}

// Loads an input 32-bit word in little-endian order from a big-endian machine.
static inline uint32_t crc64_load_le32_(const uint32_t *p) {
  uint32_t w = *p;
  return  ((((w) & 0xff000000) >> 24)
         | (((w) & 0x00ff0000) >>  8)
         | (((w) & 0x0000ff00) <<  8)
         | (((w) & 0x000000ff) << 24));
}

// A parallel multiword interleaved algorithm with a word size of 4 bytes
// and a stride factor of 5.
uint64_t crc64(const void *input, size_t nbytes) {
  const unsigned char *data = (const unsigned char*) input;
  const unsigned char *end = data + nbytes;
  uint64_t cs[5] = { UINT64_C(0xffffffffffffffff), 0, 0, 0, 0 };

  // Process byte-by-byte until proper alignment is attained.
  // In the inner loop, we process 5 4-byte words (20 bytes in total)
  // per iteration. If the amount of data remaining is small,
  // then we also use the slow algorithm.
  while (data < end && ((((size_t) data) & 3) || (end - data < 20))) {
    uint32_t idx = ((uint32_t) (cs[0] ^ *data++)) & 0xff;
    cs[0] = crc64_table[3][idx] ^ (cs[0] >> 8);
  }

  if (data == end)
    return cs[0] ^ UINT64_C(0xffffffffffffffff);

  const uint32_t one = 1;
  bool big_endian = !(*((char *)(&one)));

  uint64_t cry = 0;
  uint32_t in[5];

  if (!big_endian) {
    for (unsigned i = 0; i < 5; ++i)
      in[i] = ((const uint32_t*) data)[i];
    data += 20;

    for (; end - data >= 20; data += 20) {
      cs[0] ^= cry;

      in[0] ^= (uint32_t) cs[0];
      cs[1] ^= cs[0] >> 32;
      cs[0] = crc64_interleaved_table[0][in[0] & 0xff];
      in[0] >>= 8;

      in[1] ^= (uint32_t) cs[1];
      cs[2] ^= cs[1] >> 32;
      cs[1] = crc64_interleaved_table[0][in[1] & 0xff];
      in[1] >>= 8;

      in[2] ^= (uint32_t) cs[2];
      cs[3] ^= cs[2] >> 32;
      cs[2] = crc64_interleaved_table[0][in[2] & 0xff];
      in[2] >>= 8;

      in[3] ^= (uint32_t) cs[3];
      cs[4] ^= cs[3] >> 32;
      cs[3] = crc64_interleaved_table[0][in[3] & 0xff];
      in[3] >>= 8;

      in[4] ^= (uint32_t) cs[4];
      cry = cs[4] >> 32;
      cs[4] = crc64_interleaved_table[0][in[4] & 0xff];
      in[4] >>= 8;

      for (unsigned b = 1; b < 3; ++b) {
        cs[0] ^= crc64_interleaved_table[b][in[0] & 0xff];
        in[0] >>= 8;

        cs[1] ^= crc64_interleaved_table[b][in[1] & 0xff];
        in[1] >>= 8;

        cs[2] ^= crc64_interleaved_table[b][in[2] & 0xff];
        in[2] >>= 8;

        cs[3] ^= crc64_interleaved_table[b][in[3] & 0xff];
        in[3] >>= 8;

        cs[4] ^= crc64_interleaved_table[b][in[4] & 0xff];
        in[4] >>= 8;
      }

      cs[0] ^= crc64_interleaved_table[3][in[0] & 0xff];
      in[0] = ((const uint32_t*) data)[0];

      cs[1] ^= crc64_interleaved_table[3][in[1] & 0xff];
      in[1] = ((const uint32_t*) data)[1];

      cs[2] ^= crc64_interleaved_table[3][in[2] & 0xff];
      in[2] = ((const uint32_t*) data)[2];

      cs[3] ^= crc64_interleaved_table[3][in[3] & 0xff];
      in[3] = ((const uint32_t*) data)[3];

      cs[4] ^= crc64_interleaved_table[3][in[4] & 0xff];
      in[4] = ((const uint32_t*) data)[4];
    }
  } else {
    for (unsigned i = 0; i < 5; ++i) {
      in[i] = crc64_load_le32_(&((const uint32_t*) data)[i]);
    }
    data += 20;

    for (; end - data >= 20; data += 20) {
      cs[0] ^= cry;

      in[0] ^= (uint32_t) cs[0];
      cs[1] ^= cs[0] >> 32;
      cs[0] = crc64_interleaved_table[0][in[0] & 0xff];
      in[0] >>= 8;

      in[1] ^= (uint32_t) cs[1];
      cs[2] ^= cs[1] >> 32;
      cs[1] = crc64_interleaved_table[0][in[1] & 0xff];
      in[1] >>= 8;

      in[2] ^= (uint32_t) cs[2];
      cs[3] ^= cs[2] >> 32;
      cs[2] = crc64_interleaved_table[0][in[2] & 0xff];
      in[2] >>= 8;

      in[3] ^= (uint32_t) cs[3];
      cs[4] ^= cs[3] >> 32;
      cs[3] = crc64_interleaved_table[0][in[3] & 0xff];
      in[3] >>= 8;

      in[4] ^= (uint32_t) cs[4];
      cry = cs[4] >> 32;
      cs[4] = crc64_interleaved_table[0][in[4] & 0xff];
      in[4] >>= 8;

      for (unsigned b = 1; b < 3; ++b) {
        cs[0] ^= crc64_interleaved_table[b][in[0] & 0xff];
        in[0] >>= 8;

        cs[1] ^= crc64_interleaved_table[b][in[1] & 0xff];
        in[1] >>= 8;

        cs[2] ^= crc64_interleaved_table[b][in[2] & 0xff];
        in[2] >>= 8;

        cs[3] ^= crc64_interleaved_table[b][in[3] & 0xff];
        in[3] >>= 8;

        cs[4] ^= crc64_interleaved_table[b][in[4] & 0xff];
        in[4] >>= 8;
      }

      cs[0] ^= crc64_interleaved_table[3][in[0] & 0xff];
      in[0] = crc64_load_le32_(&((const uint32_t*) data)[0]);

      cs[1] ^= crc64_interleaved_table[3][in[1] & 0xff];
      in[1] = crc64_load_le32_(&((const uint32_t*) data)[1]);

      cs[2] ^= crc64_interleaved_table[3][in[2] & 0xff];
      in[2] = crc64_load_le32_(&((const uint32_t*) data)[2]);

      cs[3] ^= crc64_interleaved_table[3][in[3] & 0xff];
      in[3] = crc64_load_le32_(&((const uint32_t*) data)[3]);

      cs[4] ^= crc64_interleaved_table[3][in[4] & 0xff];
      in[4] = crc64_load_le32_(&((const uint32_t*) data)[4]);
    }
  }

  cs[0] ^= cry;

  for (unsigned i = 0; i < 5; ++i) {
    if (i > 0)
      cs[0] ^= cs[i];
    in[i] ^= (uint32_t) cs[0];
    cs[0] = cs[0] >> 32;

    for (unsigned b = 0; b < 3; ++b) {
      cs[0] ^= crc64_table[b][in[i] & 0xff];
      in[i] >>= 8;
    }

    cs[0] ^= crc64_table[3][in[i] & 0xff];
  }

  while (data < end) {
    uint32_t idx = ((uint32_t) (cs[0] ^ *data++)) & 0xff;
    cs[0] = crc64_table[3][idx] ^ (cs[0] >> 8);
  }

  return cs[0] ^ UINT64_C(0xffffffffffffffff);
}

inline
uint64_t crc64_device(const unsigned char *input, size_t nbytes, 
		const uint64_t *d_crc64_table, 
		const uint64_t *d_crc64_interleaved_table) {
  const unsigned char *data = input;
  const unsigned char *end = data + nbytes;
  uint64_t cs[5] = { UINT64_C(0xffffffffffffffff), 0, 0, 0, 0 };

  // Process byte-by-byte until proper alignment is attained.
  // In the inner loop, we process 5 4-byte words (20 bytes in total)
  // per iteration. If the amount of data remaining is small,
  // then we also use the slow algorithm.
  while (data < end && ((((size_t) data) & 3) || (end - data < 20))) {
    uint32_t idx = ((uint32_t) (cs[0] ^ *data++)) & 0xff;
    cs[0] = d_crc64_table[3*256+idx] ^ (cs[0] >> 8);
  }

  if (data == end)
    return cs[0] ^ UINT64_C(0xffffffffffffffff);

  const uint32_t one = 1;
  bool big_endian = !(*((char *)(&one)));

  uint64_t cry = 0;
  uint32_t in[5];

  if (!big_endian) {
    for (unsigned i = 0; i < 5; ++i)
      in[i] = ((const uint32_t*) data)[i];
    data += 20;

    for (; end - data >= 20; data += 20) {
      cs[0] ^= cry;

      in[0] ^= (uint32_t) cs[0];
      cs[1] ^= cs[0] >> 32;
      cs[0] = d_crc64_interleaved_table[in[0] & 0xff];
      in[0] >>= 8;

      in[1] ^= (uint32_t) cs[1];
      cs[2] ^= cs[1] >> 32;
      cs[1] = d_crc64_interleaved_table[in[1] & 0xff];
      in[1] >>= 8;

      in[2] ^= (uint32_t) cs[2];
      cs[3] ^= cs[2] >> 32;
      cs[2] = d_crc64_interleaved_table[in[2] & 0xff];
      in[2] >>= 8;

      in[3] ^= (uint32_t) cs[3];
      cs[4] ^= cs[3] >> 32;
      cs[3] = d_crc64_interleaved_table[in[3] & 0xff];
      in[3] >>= 8;

      in[4] ^= (uint32_t) cs[4];
      cry = cs[4] >> 32;
      cs[4] = d_crc64_interleaved_table[in[4] & 0xff];
      in[4] >>= 8;

      for (unsigned b = 1; b < 3; ++b) {
        cs[0] ^= d_crc64_interleaved_table[b*256+(in[0] & 0xff)];
        in[0] >>= 8;

        cs[1] ^= d_crc64_interleaved_table[b*256+(in[1] & 0xff)];
        in[1] >>= 8;

        cs[2] ^= d_crc64_interleaved_table[b*256+(in[2] & 0xff)];
        in[2] >>= 8;

        cs[3] ^= d_crc64_interleaved_table[b*256+(in[3] & 0xff)];
        in[3] >>= 8;

        cs[4] ^= d_crc64_interleaved_table[b*256+(in[4] & 0xff)];
        in[4] >>= 8;
      }

      cs[0] ^= d_crc64_interleaved_table[3*256+(in[0] & 0xff)];
      in[0] = ((const uint32_t*) data)[0];

      cs[1] ^= d_crc64_interleaved_table[3*256+(in[1] & 0xff)];
      in[1] = ((const uint32_t*) data)[1];

      cs[2] ^= d_crc64_interleaved_table[3*256+(in[2] & 0xff)];
      in[2] = ((const uint32_t*) data)[2];

      cs[3] ^= d_crc64_interleaved_table[3*256+(in[3] & 0xff)];
      in[3] = ((const uint32_t*) data)[3];

      cs[4] ^= d_crc64_interleaved_table[3*256+(in[4] & 0xff)];
      in[4] = ((const uint32_t*) data)[4];
    }
  } else {
    for (unsigned i = 0; i < 5; ++i) {
      in[i] = crc64_load_le32_(&((const uint32_t*) data)[i]);
    }
    data += 20;

    for (; end - data >= 20; data += 20) {
      cs[0] ^= cry;

      in[0] ^= (uint32_t) cs[0];
      cs[1] ^= cs[0] >> 32;
      cs[0] = d_crc64_interleaved_table[in[0] & 0xff];
      in[0] >>= 8;

      in[1] ^= (uint32_t) cs[1];
      cs[2] ^= cs[1] >> 32;
      cs[1] = d_crc64_interleaved_table[in[1] & 0xff];
      in[1] >>= 8;

      in[2] ^= (uint32_t) cs[2];
      cs[3] ^= cs[2] >> 32;
      cs[2] = d_crc64_interleaved_table[in[2] & 0xff];
      in[2] >>= 8;

      in[3] ^= (uint32_t) cs[3];
      cs[4] ^= cs[3] >> 32;
      cs[3] = d_crc64_interleaved_table[in[3] & 0xff];
      in[3] >>= 8;

      in[4] ^= (uint32_t) cs[4];
      cry = cs[4] >> 32;
      cs[4] = d_crc64_interleaved_table[in[4] & 0xff];
      in[4] >>= 8;

      for (unsigned b = 1; b < 3; ++b) {
        cs[0] ^= d_crc64_interleaved_table[b*256+(in[0] & 0xff)];
        in[0] >>= 8;

        cs[1] ^= d_crc64_interleaved_table[b*256+(in[1] & 0xff)];
        in[1] >>= 8;

        cs[2] ^= d_crc64_interleaved_table[b*256+(in[2] & 0xff)];
        in[2] >>= 8;

        cs[3] ^= d_crc64_interleaved_table[b*256+(in[3] & 0xff)];
        in[3] >>= 8;

        cs[4] ^= d_crc64_interleaved_table[b*256+(in[4] & 0xff)];
        in[4] >>= 8;
      }

      cs[0] ^= d_crc64_interleaved_table[3*256+(in[0] & 0xff)];
      in[0] = crc64_load_le32_(&((const uint32_t*) data)[0]);

      cs[1] ^= d_crc64_interleaved_table[3*256+(in[1] & 0xff)];
      in[1] = crc64_load_le32_(&((const uint32_t*) data)[1]);

      cs[2] ^= d_crc64_interleaved_table[3*256+(in[2] & 0xff)];
      in[2] = crc64_load_le32_(&((const uint32_t*) data)[2]);

      cs[3] ^= d_crc64_interleaved_table[3*256+(in[3] & 0xff)];
      in[3] = crc64_load_le32_(&((const uint32_t*) data)[3]);

      cs[4] ^= d_crc64_interleaved_table[3*256+(in[4] & 0xff)];
      in[4] = crc64_load_le32_(&((const uint32_t*) data)[4]);
    }
  }

  cs[0] ^= cry;

  for (unsigned i = 0; i < 5; ++i) {
    if (i > 0)
      cs[0] ^= cs[i];
    in[i] ^= (uint32_t) cs[0];
    cs[0] = cs[0] >> 32;

    for (unsigned b = 0; b < 3; ++b) {
      cs[0] ^= d_crc64_table[b*256+(in[i] & 0xff)];
      in[i] >>= 8;
    }

    cs[0] ^= d_crc64_table[3*256+(in[i] & 0xff)];
  }

  while (data < end) {
    uint32_t idx = ((uint32_t) (cs[0] ^ *data++)) & 0xff;
    cs[0] = d_crc64_table[3*256+idx] ^ (cs[0] >> 8);
  }

  return cs[0] ^ UINT64_C(0xffffffffffffffff);
}

// Calculate the 'check bytes' for the provided checksum. If these bytes are
// appended to the original buffer, then the new total checksum should be -1.
void crc64_invert(uint64_t cs, void *check_bytes) {
  unsigned char *bytes = (unsigned char *) check_bytes;
  cs ^= UINT64_C(0xffffffffffffffff);

  // The CRC is self-inverting (in big-endian, so the bit-reversed CRC is
  // self-inverting in little-endian).
  bytes[7] = (cs >> 56) & 0xff;
  bytes[6] = (cs >> 48) & 0xff;
  bytes[5] = (cs >> 40) & 0xff;
  bytes[4] = (cs >> 32) & 0xff;
  bytes[3] = (cs >> 24) & 0xff;
  bytes[2] = (cs >> 16) & 0xff;
  bytes[1] = (cs >>  8) & 0xff;
  bytes[0] =  cs        & 0xff;
}

static const uint64_t crc64_x_pow_2n[64] = {
  UINT64_C(0x4000000000000000), UINT64_C(0x2000000000000000),
  UINT64_C(0x0800000000000000), UINT64_C(0x0080000000000000),
  UINT64_C(0x0000800000000000), UINT64_C(0x0000000080000000),
  UINT64_C(0xc96c5795d7870f42), UINT64_C(0x6d5f4ad7e3c3afa0),
  UINT64_C(0xd49f7e445077d8ea), UINT64_C(0x040fb02a53c216fa),
  UINT64_C(0x6bec35957b9ef3a0), UINT64_C(0xb0e3bb0658964afe),
  UINT64_C(0x218578c7a2dff638), UINT64_C(0x6dbb920f24dd5cf2),
  UINT64_C(0x7a140cfcdb4d5eb5), UINT64_C(0x41b3705ecbc4057b),
  UINT64_C(0xd46ab656accac1ea), UINT64_C(0x329beda6fc34fb73),
  UINT64_C(0x51a4fcd4350b9797), UINT64_C(0x314fa85637efae9d),
  UINT64_C(0xacf27e9a1518d512), UINT64_C(0xffe2a3388a4d8ce7),
  UINT64_C(0x48b9697e60cc2e4e), UINT64_C(0xada73cb78dd62460),
  UINT64_C(0x3ea5454d8ce5c1bb), UINT64_C(0x5e84e3a6c70feaf1),
  UINT64_C(0x90fd49b66cbd81d1), UINT64_C(0xe2943e0c1db254e8),
  UINT64_C(0xecfa6adeca8834a1), UINT64_C(0xf513e212593ee321),
  UINT64_C(0xf36ae57331040916), UINT64_C(0x63fbd333b87b6717),
  UINT64_C(0xbd60f8e152f50b8b), UINT64_C(0xa5ce4a8299c1567d),
  UINT64_C(0x0bd445f0cbdb55ee), UINT64_C(0xfdd6824e20134285),
  UINT64_C(0xcead8b6ebda2227a), UINT64_C(0xe44b17e4f5d4fb5c),
  UINT64_C(0x9b29c81ad01ca7c5), UINT64_C(0x1b4366e40fea4055),
  UINT64_C(0x27bca1551aae167b), UINT64_C(0xaa57bcd1b39a5690),
  UINT64_C(0xd7fce83fa1234db9), UINT64_C(0xcce4986efea3ff8e),
  UINT64_C(0x3602a4d9e65341f1), UINT64_C(0x722b1da2df516145),
  UINT64_C(0xecfc3ddd3a08da83), UINT64_C(0x0fb96dcca83507e6),
  UINT64_C(0x125f2fe78d70f080), UINT64_C(0x842f50b7651aa516),
  UINT64_C(0x09bc34188cd9836f), UINT64_C(0xf43666c84196d909),
  UINT64_C(0xb56feb30c0df6ccb), UINT64_C(0xaa66e04ce7f30958),
  UINT64_C(0xb7b1187e9af29547), UINT64_C(0x113255f8476495de),
  UINT64_C(0x8fb19f783095d77e), UINT64_C(0xaec4aacc7c82b133),
  UINT64_C(0xf64e6d09218428cf), UINT64_C(0x036a72ea5ac258a0),
  UINT64_C(0x5235ef12eb7aaa6a), UINT64_C(0x2fed7b1685657853),
  UINT64_C(0x8ef8951d46606fb5), UINT64_C(0x9d58c1090f034d14)
};

// Compute (a*b) mod P
// See: https://code.google.com/p/crcutil/source/browse/code/gf_util.h
static inline uint64_t crc64_multiply_(uint64_t a, uint64_t b) {
  if ((a ^ (a-1)) < (b ^ (b-1))) {
    uint64_t t = a;
    a = b;
    b = t;
  }

  if (a == 0)
    return 0;

  uint64_t r = 0, h = UINT64_C(1) << 63;
  for (; a != 0; a <<= 1) {
    if (a & h) {
      r ^= b;
      a ^= h;
    }

    b = (b >> 1) ^ ((b & 1) ? crc64_poly : 0);
  }

  return r;
}

// Compute x**n mod P
static inline uint64_t crc64_x_pow_n_(uint64_t n) {
  uint64_t r = UINT64_C(1) << 63;
  for (size_t i = 0; n != 0; n >>= 1, ++i) {
    if (n & 1)
      r = crc64_multiply_(r, crc64_x_pow_2n[i]);
  }

  return r;
}

uint64_t crc64_combine(uint64_t cs1, uint64_t cs2, size_t nbytes2) {
  // For M = CONCAT(M1, M2) => CRC(M, a) = CRC(M2, CRC(M1, a)) and:
  // CRC(M, b) = CRC(M, a) + ((b-a)x^|M|) mod P.
  return cs2 ^ crc64_multiply_(cs1, crc64_x_pow_n_(8*nbytes2));
}

static const size_t crc64_min_thread_bytes = 1024;

uint64_t crc64_parallel(sycl::queue &q, const void *input, size_t nbytes) {

  if (nbytes > 2*crc64_min_thread_bytes) {
    int nthreads = 96*8*32;

    if (nbytes < nthreads*crc64_min_thread_bytes)
      nthreads = nbytes/crc64_min_thread_bytes;

    uint64_t thread_cs[nthreads];
    size_t thread_sz[nthreads];

    const unsigned char *data = (const unsigned char*) input;

    uint64_t *d_thread_sz = sycl::malloc_device<size_t>(nthreads, q);
    size_t *d_thread_cs = sycl::malloc_device<uint64_t>(nthreads, q);

    unsigned char *d_data = sycl::malloc_device<unsigned char>(nbytes, q);
    q.memcpy(d_data, data, nbytes);

    uint64_t *d_crc64_table = sycl::malloc_device<uint64_t>(4*256, q);
    q.memcpy(d_crc64_table, crc64_table_1D, sizeof(uint64_t) * 4 * 256);

    uint64_t *d_crc64_interleaved_table = sycl::malloc_device<uint64_t>(4*256, q);
    q.memcpy(d_crc64_interleaved_table, crc64_interleaved_table_1D, sizeof(uint64_t) * 4 * 256);

    sycl::range<1> local_size(64);
    sycl::range<1> global_size(nthreads);

    q.submit([&](sycl::handler &h) {
      h.parallel_for<class crc64_block>(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
          size_t bpt = nbytes/nthreads;
          const unsigned char *start = d_data + bpt*tid;
          const unsigned char *end;
          if (tid != nthreads - 1)
            end = start + bpt;
          else
            end = d_data + nbytes;
    
          size_t sz = end - start;
          d_thread_sz[tid] = sz;
          d_thread_cs[tid] = crc64_device(start, sz, d_crc64_table, d_crc64_interleaved_table);
      });
    });

    q.memcpy(thread_sz, d_thread_sz, sizeof(size_t) * nthreads);
    q.memcpy(thread_cs, d_thread_cs, sizeof(uint64_t) * nthreads);

    q.wait();

    uint64_t cs = thread_cs[0];
    for (int i = 1; i < nthreads; ++i) {
      cs = crc64_combine(cs, thread_cs[i], thread_sz[i]);
    }

    sycl::free(d_thread_sz, q);
    sycl::free(d_thread_cs, q);
    sycl::free(d_data, q);
    sycl::free(d_crc64_table, q);
    sycl::free(d_crc64_interleaved_table, q);
    return cs;
  }

  return crc64(input, nbytes);
}
