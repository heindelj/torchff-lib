#ifndef TORCHFF_FFTERMS_TERMS_CUH
#define TORCHFF_FFTERMS_TERMS_CUH

// Per-element energy + analytic gradient for each term, templated on the scalar type T so
// that T = Dual<scalar_t> (common/dual.cuh) turns the gradient into a Hessian-vector
// product. Everything in here must be written with the d_* math helpers and with T-typed
// constants (use `T(2)`, not `2.0`), or the Dual instantiation will not compile.
//
// Conventions: displacements are dr = r_j - r_i (torchff pair order pairs[:,0]=i, [:,1]=j),
// and each function returns the gradient with respect to *dr* plus the parameters. The
// kernel scatters dr-gradients as -g to atom i and +g to atom j.

#include "common/dual.cuh"

// ---------------------------------------------------------------------------------------
// Tang-Toennies f_6 and its derivative, cancellation-free below u = TT_SERIES_BELOW.
// f_6(u) = 1 - e^{-u} sum_{k<=6} u^k/k!   (direct, u >= 2)
//        = e^{-u} sum_{k=7}^{26} u^k/k!     (tail series, u < 2; identical algebraically)
// f_6'(u) = e^{-u} u^6 / 720 in both branches (the series telescopes to it exactly).
// ---------------------------------------------------------------------------------------
#define TT_SERIES_BELOW 2.0
#define TT_SERIES_TERMS 20

template <typename S>
__device__ __forceinline__ S inv_factorial(int k) {
    S f = S(1);
    for (int i = 2; i <= k; ++i) f *= S(i);
    return S(1) / f;
}

template <typename T, typename S>
__device__ __forceinline__ void tt_f6(T u, T& f6, T& df6) {
    T expu = d_exp(-u);
    T u2 = u * u;
    T u6 = u2 * u2 * u2;
    df6 = expu * u6 * S(1.0 / 720.0);
    if (u < S(TT_SERIES_BELOW)) {
        // Horner on the tail sum_{k=7}^{26} u^k/k! = u^7 * sum_{m=0}^{19} u^m/(m+7)!
        T poly = T(inv_factorial<S>(7 + TT_SERIES_TERMS - 1));
        for (int k = 7 + TT_SERIES_TERMS - 2; k >= 7; --k) {
            poly = poly * u + inv_factorial<S>(k);
        }
        f6 = expu * poly * u6 * u;
    } else {
        T poly = T(S(1.0 / 720.0));
        poly = poly * u + S(1.0 / 120.0);
        poly = poly * u + S(1.0 / 24.0);
        poly = poly * u + S(1.0 / 6.0);
        poly = poly * u + S(0.5);
        poly = poly * u + S(1.0);
        poly = poly * u + S(1.0);
        f6 = S(1) - expu * poly;
    }
}

// E = -f_6(b r) c6 / r^6.  grad: [d/ddr_x, d/ddr_y, d/ddr_z, d/dc6, d/db]
template <typename T, typename S>
__device__ __forceinline__ void tt_dispersion_term(
    T dx, T dy, T dz, T c6, T b, T& e, T* grad
) {
    T r2 = dx * dx + dy * dy + dz * dz;
    T r = d_sqrt(r2);
    T rinv = S(1) / r;
    T rinv2 = rinv * rinv;
    T rinv6 = rinv2 * rinv2 * rinv2;
    T u = b * r;
    T f6, df6;
    tt_f6<T, S>(u, f6, df6);
    e = -f6 * c6 * rinv6;
    // dE/dr = -c6 [ df6 * b / r^6 - 6 f6 / r^7 ]
    T dedr = -c6 * (df6 * b * rinv6 - S(6) * f6 * rinv6 * rinv);
    T s = dedr * rinv;
    grad[0] = s * dx;
    grad[1] = s * dy;
    grad[2] = s * dz;
    grad[3] = -f6 * rinv6;                 // d/dc6
    grad[4] = -c6 * df6 * r * rinv6;       // d/db
}

// Well-referenced Morse: E = D [(1 - e^{-beta (r - r_eq)})^2 - 1], beta = sqrt(k / 2D)
// grad: [d/ddr_x, d/ddr_y, d/ddr_z, d/dr_eq, d/dD, d/dk]
template <typename T, typename S>
__device__ __forceinline__ void morse_term(
    T dx, T dy, T dz, T r_eq, T dd, T k, T& e, T* grad
) {
    T r = d_sqrt(dx * dx + dy * dy + dz * dz);
    T beta = d_sqrt(k / (S(2) * dd));
    T s = r - r_eq;
    T y = d_exp(-beta * s);
    T x = S(1) - y;
    e = dd * (x * x - S(1));
    T dedr = S(2) * dd * x * beta * y;
    T q = dedr / r;
    grad[0] = q * dx;
    grad[1] = q * dy;
    grad[2] = q * dz;
    grad[3] = -dedr;                                   // d/dr_eq
    grad[4] = x * x - S(1) - x * s * y * beta;         // d/dD  (beta depends on D)
    grad[5] = dd * x * s * y * beta / k;               // d/dk  (beta depends on k)
}

// Cosine-harmonic angle: E = k/2 (cos theta - cos_eq)^2 with v1 = r_i - r_apex, v2 = r_k - r_apex
// grad: [d/dv1 (3), d/dv2 (3), d/dcos_eq, d/dk]
template <typename T, typename S>
__device__ __forceinline__ void cosine_angle_term(
    T v1x, T v1y, T v1z, T v2x, T v2y, T v2z, T cos_eq, T k, T& e, T* grad
) {
    T n1 = d_sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    T n2 = d_sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
    T inv = S(1) / (n1 * n2);
    T dot = v1x * v2x + v1y * v2y + v1z * v2z;
    T c = dot * inv;
    T diff = c - cos_eq;
    e = S(0.5) * k * diff * diff;
    T dedc = k * diff;
    T a1 = dedc * inv;                 // multiplies v2 in dc/dv1
    T b1 = dedc * c / (n1 * n1);       // multiplies v1 in dc/dv1
    T b2 = dedc * c / (n2 * n2);
    grad[0] = a1 * v2x - b1 * v1x;
    grad[1] = a1 * v2y - b1 * v1y;
    grad[2] = a1 * v2z - b1 * v1z;
    grad[3] = a1 * v1x - b2 * v2x;
    grad[4] = a1 * v1y - b2 * v2y;
    grad[5] = a1 * v1z - b2 * v2z;
    grad[6] = -dedc;                   // d/dcos_eq
    grad[7] = S(0.5) * diff * diff;    // d/dk
}

#endif /* TORCHFF_FFTERMS_TERMS_CUH */
