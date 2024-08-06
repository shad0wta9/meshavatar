#ifndef __HEMATH_H__
#define __HEMATH_H__

#include "DTypes.h"
#include <vector>

namespace HEMath {
    class Vec3;
    HEfloat inner_prod(const Vec3 & u, const Vec3 & v);
    Vec3 cross_prod(const Vec3 & u, const Vec3 & v);
};

class HEMath::Vec3 {
    HEfloat x = 0.0f;
    HEfloat y = 0.0f;
    HEfloat z = 0.0f;
  public:
    Vec3();
    Vec3(const Vec3 & other);
    Vec3(HEfloat X, HEfloat Y, HEfloat Z);
    Vec3 operator+(const Vec3 & other) const;
    Vec3 operator-(const Vec3 & other) const;
    Vec3 operator-() const;   // unary
    Vec3 operator*(HEfloat other) const;   // scalar
    Vec3 operator=(const Vec3 & other);
    Vec3 operator=(const HEfloat & other);
    bool operator==(const Vec3 & other) const;
    HEfloat operator[](int i) const;
    HEfloat & operator[](int i);

    HEfloat norm() const;
    Vec3 vertical() const;
    Vec3 normalize() const;
    void normalize_();
    friend HEfloat inner_prod(const Vec3 & v, const Vec3 & w);
    friend Vec3 cross_prod(const Vec3 & v, const Vec3 & w);

    void print();
};


#endif