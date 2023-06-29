/*
 * Copyright (c) 2009, Shanghai Jiao Tong University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 * - Neither the name of the Shanghai Jiao Tong University nor the
 *   names of its contributors may be used to endorse or promote
 *   products derived from this software without specific prior
 *   written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <sycl/sycl.hpp>

typedef unsigned char uint8_t;

static const uint8_t sbox[16] = {
  0xC0, 0x50, 0x60, 0xB0, 0x90, 0x00, 0xA0, 0xD0, 0x30, 0xE0, 0xF0, 0x80, 0x40, 0x70, 0x10, 0x20,
};

// look-up tables for speeding up permutation layer
static const uint8_t sbox_pmt_3[256] = {
  0xF0, 0xB1, 0xB4, 0xE5, 0xE1, 0xA0, 0xE4, 0xF1, 0xA5, 0xF4, 0xF5, 0xE0, 0xB0, 0xB5, 0xA1, 0xA4,
  0x72, 0x33, 0x36, 0x67, 0x63, 0x22, 0x66, 0x73, 0x27, 0x76, 0x77, 0x62, 0x32, 0x37, 0x23, 0x26,
  0x78, 0x39, 0x3C, 0x6D, 0x69, 0x28, 0x6C, 0x79, 0x2D, 0x7C, 0x7D, 0x68, 0x38, 0x3D, 0x29, 0x2C,
  0xDA, 0x9B, 0x9E, 0xCF, 0xCB, 0x8A, 0xCE, 0xDB, 0x8F, 0xDE, 0xDF, 0xCA, 0x9A, 0x9F, 0x8B, 0x8E,
  0xD2, 0x93, 0x96, 0xC7, 0xC3, 0x82, 0xC6, 0xD3, 0x87, 0xD6, 0xD7, 0xC2, 0x92, 0x97, 0x83, 0x86,
  0x50, 0x11, 0x14, 0x45, 0x41, 0x00, 0x44, 0x51, 0x05, 0x54, 0x55, 0x40, 0x10, 0x15, 0x01, 0x04,
  0xD8, 0x99, 0x9C, 0xCD, 0xC9, 0x88, 0xCC, 0xD9, 0x8D, 0xDC, 0xDD, 0xC8, 0x98, 0x9D, 0x89, 0x8C,
  0xF2, 0xB3, 0xB6, 0xE7, 0xE3, 0xA2, 0xE6, 0xF3, 0xA7, 0xF6, 0xF7, 0xE2, 0xB2, 0xB7, 0xA3, 0xA6,
  0x5A, 0x1B, 0x1E, 0x4F, 0x4B, 0x0A, 0x4E, 0x5B, 0x0F, 0x5E, 0x5F, 0x4A, 0x1A, 0x1F, 0x0B, 0x0E,
  0xF8, 0xB9, 0xBC, 0xED, 0xE9, 0xA8, 0xEC, 0xF9, 0xAD, 0xFC, 0xFD, 0xE8, 0xB8, 0xBD, 0xA9, 0xAC,
  0xFA, 0xBB, 0xBE, 0xEF, 0xEB, 0xAA, 0xEE, 0xFB, 0xAF, 0xFE, 0xFF, 0xEA, 0xBA, 0xBF, 0xAB, 0xAE,
  0xD0, 0x91, 0x94, 0xC5, 0xC1, 0x80, 0xC4, 0xD1, 0x85, 0xD4, 0xD5, 0xC0, 0x90, 0x95, 0x81, 0x84,
  0x70, 0x31, 0x34, 0x65, 0x61, 0x20, 0x64, 0x71, 0x25, 0x74, 0x75, 0x60, 0x30, 0x35, 0x21, 0x24,
  0x7A, 0x3B, 0x3E, 0x6F, 0x6B, 0x2A, 0x6E, 0x7B, 0x2F, 0x7E, 0x7F, 0x6A, 0x3A, 0x3F, 0x2B, 0x2E,
  0x52, 0x13, 0x16, 0x47, 0x43, 0x02, 0x46, 0x53, 0x07, 0x56, 0x57, 0x42, 0x12, 0x17, 0x03, 0x06,
  0x58, 0x19, 0x1C, 0x4D, 0x49, 0x08, 0x4C, 0x59, 0x0D, 0x5C, 0x5D, 0x48, 0x18, 0x1D, 0x09, 0x0C,
};

static const uint8_t sbox_pmt_2[256] = {
  0x3C, 0x6C, 0x2D, 0x79, 0x78, 0x28, 0x39, 0x7C, 0x69, 0x3D, 0x7D, 0x38, 0x2C, 0x6D, 0x68, 0x29,
  0x9C, 0xCC, 0x8D, 0xD9, 0xD8, 0x88, 0x99, 0xDC, 0xC9, 0x9D, 0xDD, 0x98, 0x8C, 0xCD, 0xC8, 0x89,
  0x1E, 0x4E, 0x0F, 0x5B, 0x5A, 0x0A, 0x1B, 0x5E, 0x4B, 0x1F, 0x5F, 0x1A, 0x0E, 0x4F, 0x4A, 0x0B,
  0xB6, 0xE6, 0xA7, 0xF3, 0xF2, 0xA2, 0xB3, 0xF6, 0xE3, 0xB7, 0xF7, 0xB2, 0xA6, 0xE7, 0xE2, 0xA3,
  0xB4, 0xE4, 0xA5, 0xF1, 0xF0, 0xA0, 0xB1, 0xF4, 0xE1, 0xB5, 0xF5, 0xB0, 0xA4, 0xE5, 0xE0, 0xA1,
  0x14, 0x44, 0x05, 0x51, 0x50, 0x00, 0x11, 0x54, 0x41, 0x15, 0x55, 0x10, 0x04, 0x45, 0x40, 0x01,
  0x36, 0x66, 0x27, 0x73, 0x72, 0x22, 0x33, 0x76, 0x63, 0x37, 0x77, 0x32, 0x26, 0x67, 0x62, 0x23,
  0xBC, 0xEC, 0xAD, 0xF9, 0xF8, 0xA8, 0xB9, 0xFC, 0xE9, 0xBD, 0xFD, 0xB8, 0xAC, 0xED, 0xE8, 0xA9,
  0x96, 0xC6, 0x87, 0xD3, 0xD2, 0x82, 0x93, 0xD6, 0xC3, 0x97, 0xD7, 0x92, 0x86, 0xC7, 0xC2, 0x83,
  0x3E, 0x6E, 0x2F, 0x7B, 0x7A, 0x2A, 0x3B, 0x7E, 0x6B, 0x3F, 0x7F, 0x3A, 0x2E, 0x6F, 0x6A, 0x2B,
  0xBE, 0xEE, 0xAF, 0xFB, 0xFA, 0xAA, 0xBB, 0xFE, 0xEB, 0xBF, 0xFF, 0xBA, 0xAE, 0xEF, 0xEA, 0xAB,
  0x34, 0x64, 0x25, 0x71, 0x70, 0x20, 0x31, 0x74, 0x61, 0x35, 0x75, 0x30, 0x24, 0x65, 0x60, 0x21,
  0x1C, 0x4C, 0x0D, 0x59, 0x58, 0x08, 0x19, 0x5C, 0x49, 0x1D, 0x5D, 0x18, 0x0C, 0x4D, 0x48, 0x09,
  0x9E, 0xCE, 0x8F, 0xDB, 0xDA, 0x8A, 0x9B, 0xDE, 0xCB, 0x9F, 0xDF, 0x9A, 0x8E, 0xCF, 0xCA, 0x8B,
  0x94, 0xC4, 0x85, 0xD1, 0xD0, 0x80, 0x91, 0xD4, 0xC1, 0x95, 0xD5, 0x90, 0x84, 0xC5, 0xC0, 0x81,
  0x16, 0x46, 0x07, 0x53, 0x52, 0x02, 0x13, 0x56, 0x43, 0x17, 0x57, 0x12, 0x06, 0x47, 0x42, 0x03,
};

static const uint8_t sbox_pmt_1[256] = {
  0x0F, 0x1B, 0x4B, 0x5E, 0x1E, 0x0A, 0x4E, 0x1F, 0x5A, 0x4F, 0x5F, 0x0E, 0x0B, 0x5B, 0x1A, 0x4A,
  0x27, 0x33, 0x63, 0x76, 0x36, 0x22, 0x66, 0x37, 0x72, 0x67, 0x77, 0x26, 0x23, 0x73, 0x32, 0x62,
  0x87, 0x93, 0xC3, 0xD6, 0x96, 0x82, 0xC6, 0x97, 0xD2, 0xC7, 0xD7, 0x86, 0x83, 0xD3, 0x92, 0xC2,
  0xAD, 0xB9, 0xE9, 0xFC, 0xBC, 0xA8, 0xEC, 0xBD, 0xF8, 0xED, 0xFD, 0xAC, 0xA9, 0xF9, 0xB8, 0xE8,
  0x2D, 0x39, 0x69, 0x7C, 0x3C, 0x28, 0x6C, 0x3D, 0x78, 0x6D, 0x7D, 0x2C, 0x29, 0x79, 0x38, 0x68,
  0x05, 0x11, 0x41, 0x54, 0x14, 0x00, 0x44, 0x15, 0x50, 0x45, 0x55, 0x04, 0x01, 0x51, 0x10, 0x40,
  0x8D, 0x99, 0xC9, 0xDC, 0x9C, 0x88, 0xCC, 0x9D, 0xD8, 0xCD, 0xDD, 0x8C, 0x89, 0xD9, 0x98, 0xC8,
  0x2F, 0x3B, 0x6B, 0x7E, 0x3E, 0x2A, 0x6E, 0x3F, 0x7A, 0x6F, 0x7F, 0x2E, 0x2B, 0x7B, 0x3A, 0x6A,
  0xA5, 0xB1, 0xE1, 0xF4, 0xB4, 0xA0, 0xE4, 0xB5, 0xF0, 0xE5, 0xF5, 0xA4, 0xA1, 0xF1, 0xB0, 0xE0,
  0x8F, 0x9B, 0xCB, 0xDE, 0x9E, 0x8A, 0xCE, 0x9F, 0xDA, 0xCF, 0xDF, 0x8E, 0x8B, 0xDB, 0x9A, 0xCA,
  0xAF, 0xBB, 0xEB, 0xFE, 0xBE, 0xAA, 0xEE, 0xBF, 0xFA, 0xEF, 0xFF, 0xAE, 0xAB, 0xFB, 0xBA, 0xEA,
  0x0D, 0x19, 0x49, 0x5C, 0x1C, 0x08, 0x4C, 0x1D, 0x58, 0x4D, 0x5D, 0x0C, 0x09, 0x59, 0x18, 0x48,
  0x07, 0x13, 0x43, 0x56, 0x16, 0x02, 0x46, 0x17, 0x52, 0x47, 0x57, 0x06, 0x03, 0x53, 0x12, 0x42,
  0xA7, 0xB3, 0xE3, 0xF6, 0xB6, 0xA2, 0xE6, 0xB7, 0xF2, 0xE7, 0xF7, 0xA6, 0xA3, 0xF3, 0xB2, 0xE2,
  0x25, 0x31, 0x61, 0x74, 0x34, 0x20, 0x64, 0x35, 0x70, 0x65, 0x75, 0x24, 0x21, 0x71, 0x30, 0x60,
  0x85, 0x91, 0xC1, 0xD4, 0x94, 0x80, 0xC4, 0x95, 0xD0, 0xC5, 0xD5, 0x84, 0x81, 0xD1, 0x90, 0xC0,
};

static const uint8_t sbox_pmt_0[256] = {
  0xC3, 0xC6, 0xD2, 0x97, 0x87, 0x82, 0x93, 0xC7, 0x96, 0xD3, 0xD7, 0x83, 0xC2, 0xD6, 0x86, 0x92,
  0xC9, 0xCC, 0xD8, 0x9D, 0x8D, 0x88, 0x99, 0xCD, 0x9C, 0xD9, 0xDD, 0x89, 0xC8, 0xDC, 0x8C, 0x98,
  0xE1, 0xE4, 0xF0, 0xB5, 0xA5, 0xA0, 0xB1, 0xE5, 0xB4, 0xF1, 0xF5, 0xA1, 0xE0, 0xF4, 0xA4, 0xB0,
  0x6B, 0x6E, 0x7A, 0x3F, 0x2F, 0x2A, 0x3B, 0x6F, 0x3E, 0x7B, 0x7F, 0x2B, 0x6A, 0x7E, 0x2E, 0x3A,
  0x4B, 0x4E, 0x5A, 0x1F, 0x0F, 0x0A, 0x1B, 0x4F, 0x1E, 0x5B, 0x5F, 0x0B, 0x4A, 0x5E, 0x0E, 0x1A,
  0x41, 0x44, 0x50, 0x15, 0x05, 0x00, 0x11, 0x45, 0x14, 0x51, 0x55, 0x01, 0x40, 0x54, 0x04, 0x10,
  0x63, 0x66, 0x72, 0x37, 0x27, 0x22, 0x33, 0x67, 0x36, 0x73, 0x77, 0x23, 0x62, 0x76, 0x26, 0x32,
  0xCB, 0xCE, 0xDA, 0x9F, 0x8F, 0x8A, 0x9B, 0xCF, 0x9E, 0xDB, 0xDF, 0x8B, 0xCA, 0xDE, 0x8E, 0x9A,
  0x69, 0x6C, 0x78, 0x3D, 0x2D, 0x28, 0x39, 0x6D, 0x3C, 0x79, 0x7D, 0x29, 0x68, 0x7C, 0x2C, 0x38,
  0xE3, 0xE6, 0xF2, 0xB7, 0xA7, 0xA2, 0xB3, 0xE7, 0xB6, 0xF3, 0xF7, 0xA3, 0xE2, 0xF6, 0xA6, 0xB2,
  0xEB, 0xEE, 0xFA, 0xBF, 0xAF, 0xAA, 0xBB, 0xEF, 0xBE, 0xFB, 0xFF, 0xAB, 0xEA, 0xFE, 0xAE, 0xBA,
  0x43, 0x46, 0x52, 0x17, 0x07, 0x02, 0x13, 0x47, 0x16, 0x53, 0x57, 0x03, 0x42, 0x56, 0x06, 0x12,
  0xC1, 0xC4, 0xD0, 0x95, 0x85, 0x80, 0x91, 0xC5, 0x94, 0xD1, 0xD5, 0x81, 0xC0, 0xD4, 0x84, 0x90,
  0xE9, 0xEC, 0xF8, 0xBD, 0xAD, 0xA8, 0xB9, 0xED, 0xBC, 0xF9, 0xFD, 0xA9, 0xE8, 0xFC, 0xAC, 0xB8,
  0x49, 0x4C, 0x58, 0x1D, 0x0D, 0x08, 0x19, 0x4D, 0x1C, 0x59, 0x5D, 0x09, 0x48, 0x5C, 0x0C, 0x18,
  0x61, 0x64, 0x70, 0x35, 0x25, 0x20, 0x31, 0x65, 0x34, 0x71, 0x75, 0x21, 0x60, 0x74, 0x24, 0x30,
};


// full-round should be 31, i.e. rounds = 31
// plain and cipher can overlap, so do key and cipher
void present_rounds(const uint8_t *plain, const uint8_t *key,
    const uint8_t rounds, uint8_t *cipher)
{
  uint8_t rounh_counter = 1;

  uint8_t state[8];
  uint8_t rounh_key[10];

  // add key
  state[0] = plain[0] ^ key[0];
  state[1] = plain[1] ^ key[1];
  state[2] = plain[2] ^ key[2];
  state[3] = plain[3] ^ key[3];
  state[4] = plain[4] ^ key[4];
  state[5] = plain[5] ^ key[5];
  state[6] = plain[6] ^ key[6];
  state[7] = plain[7] ^ key[7];

  // update key
  rounh_key[9] = key[6] << 5 | key[7] >> 3;
  rounh_key[8] = key[5] << 5 | key[6] >> 3;
  rounh_key[7] = key[4] << 5 | key[5] >> 3;
  rounh_key[6] = key[3] << 5 | key[4] >> 3;
  rounh_key[5] = key[2] << 5 | key[3] >> 3;
  rounh_key[4] = key[1] << 5 | key[2] >> 3;
  rounh_key[3] = key[0] << 5 | key[1] >> 3;
  rounh_key[2] = key[9] << 5 | key[0] >> 3;
  rounh_key[1] = key[8] << 5 | key[9] >> 3;
  rounh_key[0] = key[7] << 5 | key[8] >> 3;

  rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];

  rounh_key[7] ^= rounh_counter >> 1;
  rounh_key[8] ^= rounh_counter << 7;

  // substitution and permutation
  cipher[0] =
    (sbox_pmt_3[state[0]] & 0xC0) |
    (sbox_pmt_2[state[1]] & 0x30) |
    (sbox_pmt_1[state[2]] & 0x0C) |
    (sbox_pmt_0[state[3]] & 0x03);
  cipher[1] =
    (sbox_pmt_3[state[4]] & 0xC0) |
    (sbox_pmt_2[state[5]] & 0x30) |
    (sbox_pmt_1[state[6]] & 0x0C) |
    (sbox_pmt_0[state[7]] & 0x03);

  cipher[2] =
    (sbox_pmt_0[state[0]] & 0xC0) |
    (sbox_pmt_3[state[1]] & 0x30) |
    (sbox_pmt_2[state[2]] & 0x0C) |
    (sbox_pmt_1[state[3]] & 0x03);
  cipher[3] =
    (sbox_pmt_0[state[4]] & 0xC0) |
    (sbox_pmt_3[state[5]] & 0x30) |
    (sbox_pmt_2[state[6]] & 0x0C) |
    (sbox_pmt_1[state[7]] & 0x03);

  cipher[4] =
    (sbox_pmt_1[state[0]] & 0xC0) |
    (sbox_pmt_0[state[1]] & 0x30) |
    (sbox_pmt_3[state[2]] & 0x0C) |
    (sbox_pmt_2[state[3]] & 0x03);
  cipher[5] =
    (sbox_pmt_1[state[4]] & 0xC0) |
    (sbox_pmt_0[state[5]] & 0x30) |
    (sbox_pmt_3[state[6]] & 0x0C) |
    (sbox_pmt_2[state[7]] & 0x03);

  cipher[6] =
    (sbox_pmt_2[state[0]] & 0xC0) |
    (sbox_pmt_1[state[1]] & 0x30) |
    (sbox_pmt_0[state[2]] & 0x0C) |
    (sbox_pmt_3[state[3]] & 0x03);
  cipher[7] =
    (sbox_pmt_2[state[4]] & 0xC0) |
    (sbox_pmt_1[state[5]] & 0x30) |
    (sbox_pmt_0[state[6]] & 0x0C) |
    (sbox_pmt_3[state[7]] & 0x03);

  for (rounh_counter = 2; rounh_counter <= rounds; rounh_counter++) {
    state[0] = cipher[0] ^ rounh_key[0];
    state[1] = cipher[1] ^ rounh_key[1];
    state[2] = cipher[2] ^ rounh_key[2];
    state[3] = cipher[3] ^ rounh_key[3];
    state[4] = cipher[4] ^ rounh_key[4];
    state[5] = cipher[5] ^ rounh_key[5];
    state[6] = cipher[6] ^ rounh_key[6];
    state[7] = cipher[7] ^ rounh_key[7];

    cipher[0] =
      (sbox_pmt_3[state[0]] & 0xC0) |
      (sbox_pmt_2[state[1]] & 0x30) |
      (sbox_pmt_1[state[2]] & 0x0C) |
      (sbox_pmt_0[state[3]] & 0x03);
    cipher[1] =
      (sbox_pmt_3[state[4]] & 0xC0) |
      (sbox_pmt_2[state[5]] & 0x30) |
      (sbox_pmt_1[state[6]] & 0x0C) |
      (sbox_pmt_0[state[7]] & 0x03);

    cipher[2] =
      (sbox_pmt_0[state[0]] & 0xC0) |
      (sbox_pmt_3[state[1]] & 0x30) |
      (sbox_pmt_2[state[2]] & 0x0C) |
      (sbox_pmt_1[state[3]] & 0x03);
    cipher[3] =
      (sbox_pmt_0[state[4]] & 0xC0) |
      (sbox_pmt_3[state[5]] & 0x30) |
      (sbox_pmt_2[state[6]] & 0x0C) |
      (sbox_pmt_1[state[7]] & 0x03);

    cipher[4] =
      (sbox_pmt_1[state[0]] & 0xC0) |
      (sbox_pmt_0[state[1]] & 0x30) |
      (sbox_pmt_3[state[2]] & 0x0C) |
      (sbox_pmt_2[state[3]] & 0x03);
    cipher[5] =
      (sbox_pmt_1[state[4]] & 0xC0) |
      (sbox_pmt_0[state[5]] & 0x30) |
      (sbox_pmt_3[state[6]] & 0x0C) |
      (sbox_pmt_2[state[7]] & 0x03);

    cipher[6] =
      (sbox_pmt_2[state[0]] & 0xC0) |
      (sbox_pmt_1[state[1]] & 0x30) |
      (sbox_pmt_0[state[2]] & 0x0C) |
      (sbox_pmt_3[state[3]] & 0x03);
    cipher[7] =
      (sbox_pmt_2[state[4]] & 0xC0) |
      (sbox_pmt_1[state[5]] & 0x30) |
      (sbox_pmt_0[state[6]] & 0x0C) |
      (sbox_pmt_3[state[7]] & 0x03);

    rounh_key[5] ^= rounh_counter << 2; // do this first, which may be faster

    // use state[] for temporary storage
    state[2] = rounh_key[9];
    state[1] = rounh_key[8];
    state[0] = rounh_key[7];

    rounh_key[9] = rounh_key[6] << 5 | rounh_key[7] >> 3;
    rounh_key[8] = rounh_key[5] << 5 | rounh_key[6] >> 3;
    rounh_key[7] = rounh_key[4] << 5 | rounh_key[5] >> 3;
    rounh_key[6] = rounh_key[3] << 5 | rounh_key[4] >> 3;
    rounh_key[5] = rounh_key[2] << 5 | rounh_key[3] >> 3;
    rounh_key[4] = rounh_key[1] << 5 | rounh_key[2] >> 3;
    rounh_key[3] = rounh_key[0] << 5 | rounh_key[1] >> 3;
    rounh_key[2] = state[2] << 5 | rounh_key[0] >> 3;
    rounh_key[1] = state[1] << 5 | state[2] >> 3;
    rounh_key[0] = state[0] << 5 | state[1] >> 3;

    rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];
  }

  // if round is not equal to 31, then do not perform the last adding key operation
  // this can be used in constructing PRESENT based algorithm, such as MAC
  if (31 == rounds) {
    cipher[0] ^= rounh_key[0];
    cipher[1] ^= rounh_key[1];
    cipher[2] ^= rounh_key[2];
    cipher[3] ^= rounh_key[3];
    cipher[4] ^= rounh_key[4];
    cipher[5] ^= rounh_key[5];
    cipher[6] ^= rounh_key[6];
    cipher[7] ^= rounh_key[7];
  }
}

void present(
    const int num,
    const int rounds,
    const uint8_t *__restrict__ plains,
    const uint8_t *__restrict__ keys,
          uint8_t *__restrict__ ciphers,
    const uint8_t *__restrict__ sbox,
    const uint8_t *__restrict__ sbox_pmt_0,
    const uint8_t *__restrict__ sbox_pmt_1,
    const uint8_t *__restrict__ sbox_pmt_2,
    const uint8_t *__restrict__ sbox_pmt_3,
    const sycl::nd_item<1> &item)
{
  int gid = item.get_global_id(0);
  if (gid >= num) return;
  const uint8_t *plain = plains + gid * 8;
  const uint8_t *key = keys + gid * 10;
  uint8_t *cipher = ciphers + gid * 8;
  uint8_t rounh_counter = 1;

  uint8_t state[8];
  uint8_t rounh_key[10];

  // add key
  state[0] = plain[0] ^ key[0];
  state[1] = plain[1] ^ key[1];
  state[2] = plain[2] ^ key[2];
  state[3] = plain[3] ^ key[3];
  state[4] = plain[4] ^ key[4];
  state[5] = plain[5] ^ key[5];
  state[6] = plain[6] ^ key[6];
  state[7] = plain[7] ^ key[7];

  // update key
  rounh_key[9] = key[6] << 5 | key[7] >> 3;
  rounh_key[8] = key[5] << 5 | key[6] >> 3;
  rounh_key[7] = key[4] << 5 | key[5] >> 3;
  rounh_key[6] = key[3] << 5 | key[4] >> 3;
  rounh_key[5] = key[2] << 5 | key[3] >> 3;
  rounh_key[4] = key[1] << 5 | key[2] >> 3;
  rounh_key[3] = key[0] << 5 | key[1] >> 3;
  rounh_key[2] = key[9] << 5 | key[0] >> 3;
  rounh_key[1] = key[8] << 5 | key[9] >> 3;
  rounh_key[0] = key[7] << 5 | key[8] >> 3;

  rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];

  rounh_key[7] ^= rounh_counter >> 1;
  rounh_key[8] ^= rounh_counter << 7;

  // substitution and permutation
  cipher[0] =
    (sbox_pmt_3[state[0]] & 0xC0) |
    (sbox_pmt_2[state[1]] & 0x30) |
    (sbox_pmt_1[state[2]] & 0x0C) |
    (sbox_pmt_0[state[3]] & 0x03);
  cipher[1] =
    (sbox_pmt_3[state[4]] & 0xC0) |
    (sbox_pmt_2[state[5]] & 0x30) |
    (sbox_pmt_1[state[6]] & 0x0C) |
    (sbox_pmt_0[state[7]] & 0x03);

  cipher[2] =
    (sbox_pmt_0[state[0]] & 0xC0) |
    (sbox_pmt_3[state[1]] & 0x30) |
    (sbox_pmt_2[state[2]] & 0x0C) |
    (sbox_pmt_1[state[3]] & 0x03);
  cipher[3] =
    (sbox_pmt_0[state[4]] & 0xC0) |
    (sbox_pmt_3[state[5]] & 0x30) |
    (sbox_pmt_2[state[6]] & 0x0C) |
    (sbox_pmt_1[state[7]] & 0x03);

  cipher[4] =
    (sbox_pmt_1[state[0]] & 0xC0) |
    (sbox_pmt_0[state[1]] & 0x30) |
    (sbox_pmt_3[state[2]] & 0x0C) |
    (sbox_pmt_2[state[3]] & 0x03);
  cipher[5] =
    (sbox_pmt_1[state[4]] & 0xC0) |
    (sbox_pmt_0[state[5]] & 0x30) |
    (sbox_pmt_3[state[6]] & 0x0C) |
    (sbox_pmt_2[state[7]] & 0x03);

  cipher[6] =
    (sbox_pmt_2[state[0]] & 0xC0) |
    (sbox_pmt_1[state[1]] & 0x30) |
    (sbox_pmt_0[state[2]] & 0x0C) |
    (sbox_pmt_3[state[3]] & 0x03);
  cipher[7] =
    (sbox_pmt_2[state[4]] & 0xC0) |
    (sbox_pmt_1[state[5]] & 0x30) |
    (sbox_pmt_0[state[6]] & 0x0C) |
    (sbox_pmt_3[state[7]] & 0x03);

  for (rounh_counter = 2; rounh_counter <= rounds; rounh_counter++) {
    state[0] = cipher[0] ^ rounh_key[0];
    state[1] = cipher[1] ^ rounh_key[1];
    state[2] = cipher[2] ^ rounh_key[2];
    state[3] = cipher[3] ^ rounh_key[3];
    state[4] = cipher[4] ^ rounh_key[4];
    state[5] = cipher[5] ^ rounh_key[5];
    state[6] = cipher[6] ^ rounh_key[6];
    state[7] = cipher[7] ^ rounh_key[7];

    cipher[0] =
      (sbox_pmt_3[state[0]] & 0xC0) |
      (sbox_pmt_2[state[1]] & 0x30) |
      (sbox_pmt_1[state[2]] & 0x0C) |
      (sbox_pmt_0[state[3]] & 0x03);
    cipher[1] =
      (sbox_pmt_3[state[4]] & 0xC0) |
      (sbox_pmt_2[state[5]] & 0x30) |
      (sbox_pmt_1[state[6]] & 0x0C) |
      (sbox_pmt_0[state[7]] & 0x03);

    cipher[2] =
      (sbox_pmt_0[state[0]] & 0xC0) |
      (sbox_pmt_3[state[1]] & 0x30) |
      (sbox_pmt_2[state[2]] & 0x0C) |
      (sbox_pmt_1[state[3]] & 0x03);
    cipher[3] =
      (sbox_pmt_0[state[4]] & 0xC0) |
      (sbox_pmt_3[state[5]] & 0x30) |
      (sbox_pmt_2[state[6]] & 0x0C) |
      (sbox_pmt_1[state[7]] & 0x03);

    cipher[4] =
      (sbox_pmt_1[state[0]] & 0xC0) |
      (sbox_pmt_0[state[1]] & 0x30) |
      (sbox_pmt_3[state[2]] & 0x0C) |
      (sbox_pmt_2[state[3]] & 0x03);
    cipher[5] =
      (sbox_pmt_1[state[4]] & 0xC0) |
      (sbox_pmt_0[state[5]] & 0x30) |
      (sbox_pmt_3[state[6]] & 0x0C) |
      (sbox_pmt_2[state[7]] & 0x03);

    cipher[6] =
      (sbox_pmt_2[state[0]] & 0xC0) |
      (sbox_pmt_1[state[1]] & 0x30) |
      (sbox_pmt_0[state[2]] & 0x0C) |
      (sbox_pmt_3[state[3]] & 0x03);
    cipher[7] =
      (sbox_pmt_2[state[4]] & 0xC0) |
      (sbox_pmt_1[state[5]] & 0x30) |
      (sbox_pmt_0[state[6]] & 0x0C) |
      (sbox_pmt_3[state[7]] & 0x03);

    rounh_key[5] ^= rounh_counter << 2; // do this first, which may be faster

    // use state[] for temporary storage
    state[2] = rounh_key[9];
    state[1] = rounh_key[8];
    state[0] = rounh_key[7];

    rounh_key[9] = rounh_key[6] << 5 | rounh_key[7] >> 3;
    rounh_key[8] = rounh_key[5] << 5 | rounh_key[6] >> 3;
    rounh_key[7] = rounh_key[4] << 5 | rounh_key[5] >> 3;
    rounh_key[6] = rounh_key[3] << 5 | rounh_key[4] >> 3;
    rounh_key[5] = rounh_key[2] << 5 | rounh_key[3] >> 3;
    rounh_key[4] = rounh_key[1] << 5 | rounh_key[2] >> 3;
    rounh_key[3] = rounh_key[0] << 5 | rounh_key[1] >> 3;
    rounh_key[2] = state[2] << 5 | rounh_key[0] >> 3;
    rounh_key[1] = state[1] << 5 | state[2] >> 3;
    rounh_key[0] = state[0] << 5 | state[1] >> 3;

    rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];
  }

  // if round is not equal to 31, then do not perform the last adding key operation
  // this can be used in constructing PRESENT based algorithm, such as MAC
  if (31 == rounds) {
    cipher[0] ^= rounh_key[0];
    cipher[1] ^= rounh_key[1];
    cipher[2] ^= rounh_key[2];
    cipher[3] ^= rounh_key[3];
    cipher[4] ^= rounh_key[4];
    cipher[5] ^= rounh_key[5];
    cipher[6] ^= rounh_key[6];
    cipher[7] ^= rounh_key[7];
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <number of plain texts> <repeat>\n", argv[0]);
    return 1;
  }
  const int num = atoi(argv[1]); // number of plain texts
  const int repeat = atoi(argv[2]);

  uint seed = 8;
  srand(seed);

  // Initial 8-byte plain text
  std::array<uint8_t, 8> plain {'P', 'R', 'E', 'S', 'E', 'N', 'T', '\0'};

  // 80-bit key
  uint8_t key[10];

  // prepare data for offloading
  uint8_t* h_plain = (uint8_t*) malloc (sizeof(uint8_t) * 8 * num);
  uint8_t* h_key = (uint8_t*) malloc (sizeof(uint8_t) * 10 * num);
  uint8_t* h_cipher = (uint8_t*) malloc (sizeof(uint8_t) * 8 * num);

  // full rounds
  const int rounds = 31;

  for (int i = 0; i < num; i++) {
    // set a random key for each text
    for (int k = 0; k < 10; k++) key[k] = rand() % 256;
    memcpy(h_key+i*10, key, 10);

    memcpy(h_plain+i*8, plain.data(), 8);
    // shuffle the text
    shuffle (plain.begin(), plain.end(), std::default_random_engine(seed));
  }

  // use checksum for verification
  size_t h_checksum = 0;
  for (int n = 0; n <= repeat; n++) {
    for (int i = 0; i < num; i++) {
      present_rounds(h_plain+i*8, h_key+i*10, rounds, h_cipher+i*8);
      for (int k = 0; k < 8; k++) h_checksum += h_cipher[i*8+k];
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  uint8_t* d_plain;
  uint8_t* d_key;
  uint8_t* d_cipher;
  uint8_t* d_sbox;
  uint8_t* d_sbox_pmt_3;
  uint8_t* d_sbox_pmt_2;
  uint8_t* d_sbox_pmt_1;
  uint8_t* d_sbox_pmt_0;

  d_plain = (uint8_t *)sycl::malloc_device(8 * num, q);
  q.memcpy(d_plain, h_plain, 8 * num);

  d_key = (uint8_t *)sycl::malloc_device(10 * num, q);
  q.memcpy(d_key, h_key, 10 * num);

  d_cipher = (uint8_t *)sycl::malloc_device(8 * num, q);

  d_sbox = (uint8_t *)sycl::malloc_device(16, q);
  q.memcpy(d_sbox, sbox, 16);

  d_sbox_pmt_3 = (uint8_t *)sycl::malloc_device(256, q);
  q.memcpy(d_sbox_pmt_3, sbox_pmt_3, 256);

  d_sbox_pmt_2 = (uint8_t *)sycl::malloc_device(256, q);
  q.memcpy(d_sbox_pmt_2, sbox_pmt_2, 256);

  d_sbox_pmt_1 = (uint8_t *)sycl::malloc_device(256, q);
  q.memcpy(d_sbox_pmt_1, sbox_pmt_1, 256);

  d_sbox_pmt_0 = (uint8_t *)sycl::malloc_device(256, q);
  q.memcpy(d_sbox_pmt_0, sbox_pmt_0, 256);

  sycl::range<1> gws ((num + 255) / 256 * 256);
  sycl::range<1> lws (256);

  size_t d_checksum = 0;
  double time = 0.0;

  for (int n = 0; n <= repeat; n++) {
    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        present(num, rounds, d_plain, d_key, d_cipher, d_sbox, d_sbox_pmt_0,
                d_sbox_pmt_1, d_sbox_pmt_2, d_sbox_pmt_3, item);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    if (n > 0)
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    q.memcpy(h_cipher, d_cipher, num * 8).wait();
    for (int i = 0; i < num*8; i++) d_checksum += h_cipher[i];
  }
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  if (h_checksum != d_checksum)
    printf("FAIL\n");
  else
    printf("PASS\n");

  free(h_plain);
  free(h_key);
  free(h_cipher);
  sycl::free(d_plain, q);
  sycl::free(d_key, q);
  sycl::free(d_cipher, q);
  sycl::free(d_sbox, q);
  sycl::free(d_sbox_pmt_3, q);
  sycl::free(d_sbox_pmt_2, q);
  sycl::free(d_sbox_pmt_1, q);
  sycl::free(d_sbox_pmt_0, q);
}
