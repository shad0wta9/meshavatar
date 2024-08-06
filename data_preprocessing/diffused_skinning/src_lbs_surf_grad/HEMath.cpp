#include <cassert>
#include <cmath>
#include <vector>
#include <cstdio>
#include "HEMath.h"
#define NORMALIZE_EPS 1e-8f
#define HOMOGENIZE_EPS 1e-8f

HEMath::Vec3::Vec3(): x(0.0f), y(0.0f), z(0.0f) {}
HEMath::Vec3::Vec3(const HEMath::Vec3 & other): x(other.x), y(other.y), z(other.z) {}
HEMath::Vec3::Vec3(HEfloat X, HEfloat Y, HEfloat Z): x(X), y(Y), z(Z) {}
HEMath::Vec3 HEMath::Vec3::operator+(const HEMath::Vec3 & other) const { return HEMath::Vec3(x+other.x, y+other.y, z+other.z); }
HEMath::Vec3 HEMath::Vec3::operator-(const HEMath::Vec3 & other) const { return HEMath::Vec3(x-other.x, y-other.y, z-other.z); }
HEMath::Vec3 HEMath::Vec3::operator-() const { return HEMath::Vec3(-x, -y, -z); }
HEMath::Vec3 HEMath::Vec3::operator*(HEfloat other) const { return HEMath::Vec3(x * other, y * other, z * other); } // scalar
HEMath::Vec3 HEMath::Vec3::operator=(const HEMath::Vec3 & other) { x = other.x; y = other.y; z = other.z; return *this; }
HEMath::Vec3 HEMath::Vec3::operator=(const HEfloat & other) { x = y = z = other; return *this; }
bool HEMath::Vec3::operator==(const HEMath::Vec3 & other) const { return (x == other.x) && (y == other.y) && (z == other.z); }
HEfloat HEMath::Vec3::operator[](int i) const {
    assert((0 <= i) && (i <= 2));
    if (i == 0) return x;
    if (i == 1) return y;
    if (i == 2) return z;
}
HEfloat & HEMath::Vec3::operator[](int i) {
    assert((0 <= i) && (i <= 2));
    if (i == 0) return x;
    if (i == 1) return y;
    if (i == 2) return z;
}
HEfloat HEMath::Vec3::norm() const { return sqrt(HEMath::inner_prod(*this, *this)); }
HEMath::Vec3 HEMath::Vec3::vertical() const {
	Vec3 result = cross_prod(*this, Vec3(0.0f, 0.0f, 1.0f));
	if ( result.norm() < NORMALIZE_EPS )
        return Vec3(1.0f, 0.0f, 0.0f);
    return result.normalize();
}

HEMath::Vec3 HEMath::Vec3::normalize() const {
    HEfloat vec_len = std::sqrt(inner_prod(*this, *this));
    if (vec_len < NORMALIZE_EPS) {
        return Vec3(0.0f, 0.0f, 0.0f);
    }
    return Vec3(x / vec_len, y / vec_len, z / vec_len);
}

void HEMath::Vec3::normalize_() {
    HEfloat vec_len = std::sqrt(inner_prod(*this, *this));
    if (vec_len < NORMALIZE_EPS) {
        x = y = z = 0.0f;
    }
    x /= vec_len;
    y /= vec_len;
    z /= vec_len;
}
HEfloat HEMath::inner_prod(const HEMath::Vec3 & v, const HEMath::Vec3 & w) { return v.x * w.x + v.y * w.y + v.z * w.z; }    // friend function
HEMath::Vec3 HEMath::cross_prod(const HEMath::Vec3 & v, const HEMath::Vec3 & w) {   // friend function
    return HEMath::Vec3(
        v.y * w.z - v.z * w.y,
        v.z * w.x - v.x * w.z,
        v.x * w.y - v.y * w.x
    );
}
void HEMath::Vec3::print() { printf("Vec4(%f, %f, %f)", x, y, z); }
