#ifndef TORCHFF_DUAL_CUH
#define TORCHFF_DUAL_CUH

// Forward-mode dual numbers for second derivatives of per-pair terms.
//
// A gradient kernel written as a template `template <typename T> grad(T inputs..., T* grad)`
// can be instantiated with T = Dual<scalar_t>. Seeding the inputs' tangents with a direction
// v then yields grad(x) in the primal part and (d/dt) grad(x + t v) = H v in the tangent part:
// exactly the Hessian-vector product that the backward of a gradient op needs. One
// hand-written first derivative per term therefore gives double backward for free, at ~2x
// the cost of the gradient kernel.
//
// The math helpers below (d_exp, d_sqrt, ...) are what the templated per-term code must
// use, so that both instantiations compile: plain scalars route to the CUDA intrinsics,
// duals propagate the tangent through the chain rule.

#include <cuda_runtime.h>

template <typename T>
struct Dual {
    T v;  // primal value
    T d;  // tangent (directional derivative)

    __device__ __forceinline__ Dual() : v(0), d(0) {}
    __device__ __forceinline__ Dual(T v_) : v(v_), d(0) {}
    __device__ __forceinline__ Dual(T v_, T d_) : v(v_), d(d_) {}

    __device__ __forceinline__ Dual operator-() const { return Dual(-v, -d); }
    __device__ __forceinline__ Dual& operator+=(const Dual& o) { v += o.v; d += o.d; return *this; }
    __device__ __forceinline__ Dual& operator-=(const Dual& o) { v -= o.v; d -= o.d; return *this; }
    __device__ __forceinline__ Dual& operator*=(const Dual& o) { d = d * o.v + v * o.d; v *= o.v; return *this; }
};

// ---- arithmetic: Dual (op) Dual ----------------------------------------------------------
template <typename T> __device__ __forceinline__ Dual<T> operator+(const Dual<T>& a, const Dual<T>& b) { return Dual<T>(a.v + b.v, a.d + b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator-(const Dual<T>& a, const Dual<T>& b) { return Dual<T>(a.v - b.v, a.d - b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator*(const Dual<T>& a, const Dual<T>& b) { return Dual<T>(a.v * b.v, a.d * b.v + a.v * b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator/(const Dual<T>& a, const Dual<T>& b) {
    T inv = T(1) / b.v;
    return Dual<T>(a.v * inv, (a.d * b.v - a.v * b.d) * inv * inv);
}
// ---- arithmetic: Dual (op) scalar, scalar (op) Dual --------------------------------------
template <typename T> __device__ __forceinline__ Dual<T> operator+(const Dual<T>& a, T b) { return Dual<T>(a.v + b, a.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator+(T a, const Dual<T>& b) { return Dual<T>(a + b.v, b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator-(const Dual<T>& a, T b) { return Dual<T>(a.v - b, a.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator-(T a, const Dual<T>& b) { return Dual<T>(a - b.v, -b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator*(const Dual<T>& a, T b) { return Dual<T>(a.v * b, a.d * b); }
template <typename T> __device__ __forceinline__ Dual<T> operator*(T a, const Dual<T>& b) { return Dual<T>(a * b.v, a * b.d); }
template <typename T> __device__ __forceinline__ Dual<T> operator/(const Dual<T>& a, T b) { T inv = T(1) / b; return Dual<T>(a.v * inv, a.d * inv); }
template <typename T> __device__ __forceinline__ Dual<T> operator/(T a, const Dual<T>& b) { T inv = T(1) / b.v; return Dual<T>(a * inv, -a * b.d * inv * inv); }

// ---- comparisons act on the primal (branch selection) -------------------------------------
template <typename T> __device__ __forceinline__ bool operator<(const Dual<T>& a, T b) { return a.v < b; }
template <typename T> __device__ __forceinline__ bool operator>(const Dual<T>& a, T b) { return a.v > b; }

// ---- math helpers usable on both T and Dual<T> --------------------------------------------
__device__ __forceinline__ float  d_exp(float x)  { return ::expf(x); }
__device__ __forceinline__ double d_exp(double x) { return ::exp(x); }
__device__ __forceinline__ float  d_sqrt(float x)  { return ::sqrtf(x); }
__device__ __forceinline__ double d_sqrt(double x) { return ::sqrt(x); }

template <typename T> __device__ __forceinline__ Dual<T> d_exp(const Dual<T>& x) {
    T e = d_exp(x.v);
    return Dual<T>(e, e * x.d);
}
template <typename T> __device__ __forceinline__ Dual<T> d_sqrt(const Dual<T>& x) {
    T s = d_sqrt(x.v);
    return Dual<T>(s, x.d / (T(2) * s));
}

// primal(x): the value part, for branch decisions and for reading results back
__device__ __forceinline__ float  primal(float x)  { return x; }
__device__ __forceinline__ double primal(double x) { return x; }
template <typename T> __device__ __forceinline__ T primal(const Dual<T>& x) { return x.v; }

// tangent(x): zero for a plain scalar, the directional derivative for a dual
__device__ __forceinline__ float  tangent(float)  { return 0.f; }
__device__ __forceinline__ double tangent(double) { return 0.; }
template <typename T> __device__ __forceinline__ T tangent(const Dual<T>& x) { return x.d; }

#endif /* TORCHFF_DUAL_CUH */
