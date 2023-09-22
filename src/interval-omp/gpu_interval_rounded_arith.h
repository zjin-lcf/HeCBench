/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Type-specific implementation of rounded arithmetic operators.

#ifndef GPU_INTERVAL_ROUNDED_ARITH_H
#define GPU_INTERVAL_ROUNDED_ARITH_H

// Generic class, no actual implementation yet
template <class T>
struct rounded_arith {
  T add_down(const T &x, const T &y);
  T add_up(const T &x, const T &y);
  T sub_down(const T &x, const T &y);
  T sub_up(const T &x, const T &y);
  T mul_down(const T &x, const T &y);
  T mul_up(const T &x, const T &y);
  T div_down(const T &x, const T &y);
  T div_up(const T &x, const T &y);
  T median(const T &x, const T &y);
  T sqrt_down(const T &x);
  T sqrt_up(const T &x);
  T int_down(const T &x);
  T int_up(const T &x);

  T pos_inf();
  T neg_inf();
  T nan();
  T min(T const &x, T const &y);
  T max(T const &x, T const &y);
};

// Specialization for float
template <>
struct rounded_arith<float> {
  float add_down(const float &x, const float &y) {
    return x + y;
  }

  float add_up(const float &x, const float &y) {
    return x + y;
  }

  float sub_down(const float &x, const float &y) {
    return x + (-y);
  }

  float sub_up(const float &x, const float &y) {
    return x + (-y);
  }

  float mul_down(const float &x, const float &y) {
    return x * y;
  }

  float mul_up(const float &x, const float &y) {
    return x * y;
  }

  float div_down(const float &x, const float &y) {
    return x / y;
  }

  float div_up(const float &x, const float &y) {
    return x / y;
  }

  float median(const float &x, const float &y) {
    return (x + y) * .5f;
  }

  float sqrt_down(const float &x) { return sqrt((float)x); }

  float sqrt_up(const float &x) { return sqrt((float)x); }

  float int_down(const float &x) { return floor((float)x); }

  float int_up(const float &x) { return ceil((float)x); }

  float neg_inf() {
    return -std::numeric_limits<float>::infinity();
  }

  float pos_inf() {
    return std::numeric_limits<float>::infinity();
  }

  float nan() { return nanf(""); }

  float min(float const &x, float const &y) {
    return fminf((float)x, (float)y);
  }

  float max(float const &x, float const &y) {
    return fmaxf((float)x, (float)y);
  }
};

// Specialization for double
template <>
struct rounded_arith<double> {
  double add_down(const double &x, const double &y) {
    return x + y;
  }

  double add_up(const double &x, const double &y) {
    return x + y;
  }

  double sub_down(const double &x, const double &y) {
    return x + (-y);
  }

  double sub_up(const double &x, const double &y) {
    return x + (-y);
  }

  double mul_down(const double &x, const double &y) {
    return x * y;
  }

  double mul_up(const double &x, const double &y) {
    return x * y;
  }

  double div_down(const double &x, const double &y) {
    return x / y;
  }

  double div_up(const double &x, const double &y) {
    return x / y;
  }
  double median(const double &x, const double &y) {
    return (x + y) * .5;
  }

  double sqrt_down(const double &x) { return sqrt((double)x); }

  double sqrt_up(const double &x) { return sqrt((double)x); }

  double int_down(const double &x) { return floor((double)x); }

  double int_up(const double &x) { return ceil((double)x); }

  double neg_inf() {
    return -std::numeric_limits<double>::infinity();
  }

  double pos_inf() {
    return std::numeric_limits<double>::infinity();
  }
  double nan() { return ::nan(""); }

  double min(double const &x, double const &y) {
    return fmin((double)x, (double)y);
  }

  double max(double const &x, double const &y) {
    return fmax((double)x, (double)y);
  }
};

#endif
