
#ifndef _FLOAT_MATH_H
#define _FLOAT_MATH_H

#include <sycl/sycl.hpp>
#include <math.h>

// inline float fminf (const float a, const float b) { return a < b ? a : b; }
// inline float fmaxf (const float a, const float b) { return a > b ? a : b; }

inline float clamp(const float f, const float a, const float b) {
    return sycl::fmax((float)a, sycl::fmin((float)f, (float)b));
}

inline sycl::float2 make_float2(const float s) { return sycl::float2(s, 0.f); }

inline sycl::float2 operator-(const sycl::float2 a) {
    return sycl::float2(-a.x(), -a.y());
}

inline sycl::float2 operator+(const sycl::float2 a, const sycl::float2 b) {
    return sycl::float2(a.x() + b.x(), a.y() + b.y());
}

inline sycl::float2 operator+(const sycl::float2 a, const float b) {
    return sycl::float2(a.x() + b, a.y());
}

inline sycl::float2 operator+(const float a, const sycl::float2 b) {
    return sycl::float2(a + b.x(), b.y());
}

inline void operator+=(sycl::float2 &a, const sycl::float2 b) {
    a.x() += b.x(); a.y() += b.y();
}

inline sycl::float2 operator-(const sycl::float2 a, const sycl::float2 b) {
    return sycl::float2(a.x() - b.x(), a.y() - b.y());
}

inline void operator-=(sycl::float2 &a, const sycl::float2 b) {
    a.x() -= b.x(); a.y() -= b.y();
}

inline sycl::float2 operator*(const sycl::float2 a, const float s) {
    return sycl::float2(a.x() * s, a.y() * s);
}

inline sycl::float2 operator*(const float s, const sycl::float2 a) {
    return sycl::float2(a.x() * s, a.y() * s);
}

inline void operator*=(sycl::float2 &a, const float s) {
    a.x() *= s; a.y() *= s;
}

inline void operator/=(sycl::float2 &a, const float s) {
    float inv = 1.0f / s; a*=inv;
}

inline sycl::float2 operator/(const sycl::float2 a, const float s)
{
    float inv = 1.0f / s;
    return a * inv;
}

inline void operator*=(sycl::float2 &a, const sycl::float2 b)
{
    float s = a.x();
    float t = a.y();
    a.x() = s * b.x() - t * b.y();
    a.y() = t * b.x() + s * b.y();
}

inline sycl::float2 operator*(const sycl::float2 a, const sycl::float2 b)
{
    // (a.x + i a.y) * (b.x + i b.y) = a.x*b.x - a.y*b.y + i(a.y*b.x + a.x*b.y)
    return sycl::float2(a.x() * b.x() - a.y() * b.y(),
                        a.x() * b.y() + a.y() * b.x());
}

inline float dot(const sycl::float2 a, const sycl::float2 b) {
    return a.x() * b.x() + a.y() * b.y();
}
inline float norm(const sycl::float2 a) {
    return a.x() * a.x() + a.y() * a.y();
}
inline float abs(const sycl::float2 a) {
    return sycl::hypot((float)(a.x()), (float)(a.y()));
}
inline sycl::float2 conj(const sycl::float2 a) {
    return sycl::float2(a.x(), -a.y());
}

inline sycl::float2 operator/(const float s, const sycl::float2 b)
{
    return (s * conj(b)) / norm(b);
}

inline sycl::float2 operator/(const sycl::float2 a, const sycl::float2 b)
{
    return (a * conj(b)) / norm(b);
}

inline float cargf(const sycl::float2 a) {
    return sycl::atan2((float)(a.x()), (float)(a.y()));
}

inline sycl::float2 cexpf(const float a)
{
    float c, s;
    s = sycl::sincos(a, &c);
    return sycl::float2(c, s);
}

inline bool iszero(const sycl::float2 a) {
    return (a.x() == 0.f && a.y() == 0.f);
}

#endif /* _FLOAT_MATH_H */
