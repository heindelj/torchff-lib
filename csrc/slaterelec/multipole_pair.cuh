#ifndef TORCHFF_SLATER_ELEC_MULTIPOLE_PAIR_CUH
#define TORCHFF_SLATER_ELEC_MULTIPOLE_PAIR_CUH

// Damped multipole pair interaction a_j^T T(dr; damps) a_i up to quadrupoles, with the
// gradient with respect to both multipoles and to dr.
//
// GENERATED from csrc/cmm/multipoles.cuh::pairwise_multipole_kernel_with_grad (the overload
// with dr gradients) by tools/gen_multipole_pair.py: `scalar_t` -> `T`, `rsqrt_` -> `d_rsqrt`.
// Do not edit by hand -- regenerate. The template parameter T may be a plain scalar or
// Dual<scalar> (common/dual.cuh); the integer literals in the tensor algebra are handled by
// Dual's int overloads.
//
// Conventions (pyCMM): dr = r_j - r_i; polytensor slots [q, mu_x, mu_y, mu_z, Qxx, Qxy, Qxz,
// Qyy, Qyz, Qzz] with the [1/3, 2/3, 2/3, 1/3, 2/3, 1/3] weights already applied to the
// Cartesian quadrupole. damp1..damp11 scale 1/r, 1/r^3, ..., 1/r^11; the dr gradient uses
// the next order's damping, which is exact when the damps are a consistent derivative family
// (they are: tests/host/check_slater_elec.cpp checks against torch autograd).

#include "common/dual.cuh"

template <typename T>
__device__ __forceinline__ void mp_pair_grad(
    T c0_i,
    T dx_i, T dy_i, T dz_i,
    T qxx_i, T qxy_i, T qxz_i, T qyy_i, T qyz_i, T qzz_i,
    T c0_j,
    T dx_j, T dy_j, T dz_j,
    T qxx_j, T qxy_j, T qxz_j, T qyy_j, T qyz_j, T qzz_j,
    T drx, T dry, T drz,
    T damp1, T damp3, T damp5, T damp7, T damp9, T damp11,
    T* ene,
    T* c0_i_g,
    T* dx_i_g, T* dy_i_g, T* dz_i_g,
    T* qxx_i_g, T* qxy_i_g, T* qxz_i_g, T* qyy_i_g, T* qyz_i_g, T* qzz_i_g,
    T* c0_j_g,
    T* dx_j_g, T* dy_j_g, T* dz_j_g,
    T* qxx_j_g, T* qxy_j_g, T* qxz_j_g, T* qyy_j_g, T* qyz_j_g, T* qzz_j_g,
    T* drx_g, T* dry_g, T* drz_g
) 
{
    // dr = rj - ri;
    T drinv = d_rsqrt(drx*drx+dry*dry+drz*drz);
    T drinv2 = drinv * drinv;
    T drinv3 = drinv2 * drinv;
    T drinv5 = drinv3 * drinv2;
    T drinv7 = drinv5 * drinv2;
    T drinv9 = drinv7 * drinv2;
    T drinv11 = drinv9 * drinv2;

    drinv *= damp1;
    drinv3 *= damp3;
    drinv5 *= damp5;
    drinv7 *= damp7;
    drinv9 *= damp9;
    drinv11 *= damp11;

    T tx = -drx * drinv3; 
    T ty = -dry * drinv3;
    T tz = -drz * drinv3;

    T x2 = drx * drx;
    T xy = drx * dry;
    T xz = drx * drz;
    T y2 = dry * dry;
    T yz = dry * drz;
    T z2 = drz * drz;
    T xyz = drx * dry * drz;
    
    T txx = 3 * x2 * drinv5 - drinv3;
    T txy = 3 * xy * drinv5;
    T txz = 3 * xz * drinv5;
    T tyy = 3 * y2 * drinv5 - drinv3;
    T tyz = 3 * yz * drinv5;
    T tzz = 3 * z2 * drinv5 - drinv3;     

    T txxx = -15 * x2 * drx * drinv7 + 9 * drx * drinv5;
    T txxy = -15 * x2 * dry * drinv7 + 3 * dry * drinv5;
    T txxz = -15 * x2 * drz * drinv7 + 3 * drz * drinv5;
    T tyyy = -15 * y2 * dry * drinv7 + 9 * dry * drinv5;
    T tyyx = -15 * y2 * drx * drinv7 + 3 * drx * drinv5;
    T tyyz = -15 * y2 * drz * drinv7 + 3 * drz * drinv5;
    T tzzz = -15 * z2 * drz * drinv7 + 9 * drz * drinv5;
    T tzzx = -15 * z2 * drx * drinv7 + 3 * drx * drinv5;
    T tzzy = -15 * z2 * dry * drinv7 + 3 * dry * drinv5;
    T txyz = -15 * xyz * drinv7;

    T txxxx = 105 * x2 * x2 * drinv9 - 90 * x2 * drinv7 + 9 * drinv5;
    T txxxy = 105 * x2 * xy * drinv9 - 45 * xy * drinv7;
    T txxxz = 105 * x2 * xz * drinv9 - 45 * xz * drinv7;
    T txxyy = 105 * x2 * y2 * drinv9 - 15 * (x2 + y2) * drinv7 + 3 * drinv5;
    T txxzz = 105 * x2 * z2 * drinv9 - 15 * (x2 + z2) * drinv7 + 3 * drinv5;
    T txxyz = 105 * x2 * yz * drinv9 - 15 * yz * drinv7;

    T tyyyy = 105 * y2 * y2 * drinv9 - 90 * y2 * drinv7 + 9 * drinv5;
    T tyyyx = 105 * y2 * xy * drinv9 - 45 * xy * drinv7;
    T tyyyz = 105 * y2 * yz * drinv9 - 45 * yz * drinv7;
    T tyyzz = 105 * y2 * z2 * drinv9 - 15 * (y2 + z2) * drinv7 + 3 * drinv5;
    T tyyxz = 105 * y2 * xz * drinv9 - 15 * xz * drinv7;

    T tzzzz = 105 * z2 * z2 * drinv9 - 90 * z2 * drinv7 + 9 * drinv5;
    T tzzzx = 105 * z2 * xz * drinv9 - 45 * xz * drinv7;
    T tzzzy = 105 * z2 * yz * drinv9 - 45 * yz * drinv7;                
    T tzzxy = 105 * z2 * xy * drinv9 - 15 * xy * drinv7;

    // interaction tensor graident
    T c0prod = c0_i * c0_j;

    T tx_g = c0_i * dx_j - c0_j * dx_i; 
    T ty_g = c0_i * dy_j - c0_j * dy_i;
    T tz_g = c0_i * dz_j - c0_j * dz_i;
    
    T txx_g = c0_i * qxx_j + c0_j * qxx_i - dx_i * dx_j;
    T txy_g = c0_i * qxy_j + c0_j * qxy_i - dx_i * dy_j - dx_j * dy_i;
    T txz_g = c0_i * qxz_j + c0_j * qxz_i - dx_i * dz_j - dx_j * dz_i;
    T tyy_g = c0_i * qyy_j + c0_j * qyy_i - dy_i * dy_j;
    T tyz_g = c0_i * qyz_j + c0_j * qyz_i - dy_i * dz_j - dy_j * dz_i;
    T tzz_g = c0_i * qzz_j + c0_j * qzz_i - dz_i * dz_j;     

    T txxx_g = qxx_i * dx_j - qxx_j * dx_i;
    T txxy_g = qxx_i * dy_j - qxx_j * dy_i + qxy_i * dx_j - qxy_j * dx_i;
    T txxz_g = qxx_i * dz_j - qxx_j * dz_i + qxz_i * dx_j - qxz_j * dx_i;
    T tyyy_g = qyy_i * dy_j - qyy_j * dy_i;
    T tyyx_g = qyy_i * dx_j - qyy_j * dx_i + qxy_i * dy_j - qxy_j * dy_i;
    T tyyz_g = qyy_i * dz_j - qyy_j * dz_i + qyz_i * dy_j - qyz_j * dy_i;
    T tzzz_g = qzz_i * dz_j - qzz_j * dz_i;
    T tzzx_g = qzz_i * dx_j - qzz_j * dx_i + qxz_i * dz_j - qxz_j * dz_i;
    T tzzy_g = qzz_i * dy_j - qzz_j * dy_i + qyz_i * dz_j - qyz_j * dz_i;
    T txyz_g = qxy_i * dz_j - qxy_j * dz_i + qxz_i * dy_j - qxz_j * dy_i + qyz_i * dx_j - qyz_j * dx_i;

    T txxxx_g = qxx_i * qxx_j;
    T txxxy_g = qxx_i * qxy_j + qxy_i * qxx_j;
    T txxxz_g = qxx_i * qxz_j + qxz_i * qxx_j;
    T txxyy_g = qxx_i * qyy_j + qyy_i * qxx_j + qxy_i * qxy_j;
    T txxzz_g = qxx_i * qzz_j + qzz_i * qxx_j + qxz_i * qxz_j;
    T txxyz_g = qxx_i * qyz_j + qyz_i * qxx_j + qxy_i * qxz_j + qxz_i * qxy_j;

    T tyyyy_g = qyy_i * qyy_j;
    T tyyyx_g = qyy_i * qxy_j + qxy_i * qyy_j;
    T tyyyz_g = qyy_i * qyz_j + qyz_i * qyy_j;
    T tyyzz_g = qyy_i * qzz_j + qzz_i * qyy_j + qyz_i * qyz_j;
    T tyyxz_g = qyy_i * qxz_j + qxz_i * qyy_j + qxy_i * qyz_j + qyz_i * qxy_j;

    T tzzzz_g = qzz_i * qzz_j;
    T tzzzx_g = qzz_i * qxz_j + qxz_i * qzz_j;
    T tzzzy_g = qzz_i * qyz_j + qyz_i * qzz_j;                
    T tzzxy_g = qzz_i * qxy_j + qxy_i * qzz_j + qxz_i * qyz_j + qyz_i * qxz_j;

    *ene = c0prod*drinv + tx * tx_g + ty * ty_g + tz * tz_g 
        + txx * txx_g + txy * txy_g + txz * txz_g + tyy * tyy_g + tyz * tyz_g + tzz * tzz_g
        + txxx * txxx_g + txxy * txxy_g + txxz * txxz_g + tyyy * tyyy_g + tyyx * tyyx_g + tyyz * tyyz_g + tzzz * tzzz_g + tzzx * tzzx_g + tzzy * tzzy_g + txyz * txyz_g
        + txxxx * txxxx_g
        + txxxy * txxxy_g
        + txxxz * txxxz_g
        + txxyy * txxyy_g
        + txxzz * txxzz_g
        + txxyz * txxyz_g
        + tyyyy * tyyyy_g
        + tyyyx * tyyyx_g
        + tyyyz * tyyyz_g
        + tyyzz * tyyzz_g
        + tyyxz * tyyxz_g
        + tzzzz * tzzzz_g
        + tzzzx * tzzzx_g
        + tzzzy * tzzzy_g
        + tzzxy * tzzxy_g; // quad-quad
    
    // charge gradient - electric potential
    *c0_i_g = drinv * c0_j + tx * dx_j + ty * dy_j + tz * dz_j + txx * qxx_j + txy * qxy_j + txz * qxz_j + tyy * qyy_j + tyz * qyz_j + tzz * qzz_j;
    *c0_j_g = drinv * c0_i - tx * dx_i - ty * dy_i - tz * dz_i + txx * qxx_i + txy * qxy_i + txz * qxz_i + tyy * qyy_i + tyz * qyz_i + tzz * qzz_i;
    
    // dipole gradient - electric field
    *dx_i_g = -c0_j * tx - txx * dx_j - txy * dy_j - txz * dz_j - txxx * qxx_j - txxy * qxy_j - txxz * qxz_j - tyyx * qyy_j - tzzx * qzz_j - txyz * qyz_j;
    *dy_i_g = -c0_j * ty - txy * dx_j - tyy * dy_j - tyz * dz_j - txxy * qxx_j - tyyx * qxy_j - txyz * qxz_j - tyyy * qyy_j - tzzy * qzz_j - tyyz * qyz_j;
    *dz_i_g = -c0_j * tz - txz * dx_j - tyz * dy_j - tzz * dz_j - txxz * qxx_j - txyz * qxy_j - tzzx * qxz_j - tyyz * qyy_j - tzzz * qzz_j - tzzy * qyz_j;

    *dx_j_g = c0_i * tx - txx * dx_i - txy * dy_i - txz * dz_i + txxx * qxx_i + txxy * qxy_i + txxz * qxz_i + tyyx * qyy_i + tzzx * qzz_i + txyz * qyz_i;
    *dy_j_g = c0_i * ty - txy * dx_i - tyy * dy_i - tyz * dz_i + txxy * qxx_i + tyyx * qxy_i + txyz * qxz_i + tyyy * qyy_i + tzzy * qzz_i + tyyz * qyz_i;
    *dz_j_g = c0_i * tz - txz * dx_i - tyz * dy_i - tzz * dz_i + txxz * qxx_i + txyz * qxy_i + tzzx * qxz_i + tyyz * qyy_i + tzzz * qzz_i + tzzy * qyz_i;

    // quadrupole gradient - electric field graident
    *qxx_i_g = c0_j * txx + txxx * dx_j + txxy * dy_j + txxz * dz_j + txxxx * qxx_j + txxxy * qxy_j + txxxz * qxz_j + txxyy * qyy_j + txxyz * qyz_j + txxzz * qzz_j;
    *qxy_i_g = c0_j * txy + txxy * dx_j + tyyx * dy_j + txyz * dz_j + txxxy * qxx_j + txxyy * qxy_j + txxyz * qxz_j + tyyyx * qyy_j + tyyxz * qyz_j + tzzxy * qzz_j;
    *qxz_i_g = c0_j * txz + txxz * dx_j + txyz * dy_j + tzzx * dz_j + txxxz * qxx_j + txxyz * qxy_j + txxzz * qxz_j + tyyxz * qyy_j + tzzxy * qyz_j + tzzzx * qzz_j;
    *qyy_i_g = c0_j * tyy + tyyx * dx_j + tyyy * dy_j + tyyz * dz_j + txxyy * qxx_j + tyyyx * qxy_j + tyyxz * qxz_j + tyyyy * qyy_j + tyyyz * qyz_j + tyyzz * qzz_j;
    *qyz_i_g = c0_j * tyz + txyz * dx_j + tyyz * dy_j + tzzy * dz_j + txxyz * qxx_j + tyyxz * qxy_j + tzzxy * qxz_j + tyyyz * qyy_j + tyyzz * qyz_j + tzzzy * qzz_j;
    *qzz_i_g = c0_j * tzz + tzzx * dx_j + tzzy * dy_j + tzzz * dz_j + txxzz * qxx_j + tzzxy * qxy_j + tzzzx * qxz_j + tyyzz * qyy_j + tzzzy * qyz_j + tzzzz * qzz_j;

    *qxx_j_g = c0_i * txx - txxx * dx_i - txxy * dy_i - txxz * dz_i + txxxx * qxx_i + txxxy * qxy_i + txxxz * qxz_i + txxyy * qyy_i + txxyz * qyz_i + txxzz * qzz_i;
    *qxy_j_g = c0_i * txy - txxy * dx_i - tyyx * dy_i - txyz * dz_i + txxxy * qxx_i + txxyy * qxy_i + txxyz * qxz_i + tyyyx * qyy_i + tyyxz * qyz_i + tzzxy * qzz_i;
    *qxz_j_g = c0_i * txz - txxz * dx_i - txyz * dy_i - tzzx * dz_i + txxxz * qxx_i + txxyz * qxy_i + txxzz * qxz_i + tyyxz * qyy_i + tzzxy * qyz_i + tzzzx * qzz_i;
    *qyy_j_g = c0_i * tyy - tyyx * dx_i - tyyy * dy_i - tyyz * dz_i + txxyy * qxx_i + tyyyx * qxy_i + tyyxz * qxz_i + tyyyy * qyy_i + tyyyz * qyz_i + tyyzz * qzz_i;
    *qyz_j_g = c0_i * tyz - txyz * dx_i - tyyz * dy_i - tzzy * dz_i + txxyz * qxx_i + tyyxz * qxy_i + tzzxy * qxz_i + tyyyz * qyy_i + tyyzz * qyz_i + tzzzy * qzz_i;
    *qzz_j_g = c0_i * tzz - tzzx * dx_i - tzzy * dy_i - tzzz * dz_i + txxzz * qxx_i + tzzxy * qxy_i + tzzzx * qxz_i + tyyzz * qyy_i + tzzzy * qyz_i + tzzzz * qzz_i;

    // dr gradient - forces
    T c945dr11 = -945 * drinv11; 
    T c105dr9 = 105 * drinv9;
    T c15dr7 = 15 * drinv7;

    T t5x = c945dr11 * x2 * x2 * drx + 10 * c105dr9 * x2 * drx - 15 * c15dr7 * drx;
    T t5y = c945dr11 * y2 * y2 * dry + 10 * c105dr9 * y2 * dry - 15 * c15dr7 * dry;
    T t5z = c945dr11 * z2 * z2 * drz + 10 * c105dr9 * z2 * drz - 15 * c15dr7 * drz;
    T t4x1y = c945dr11 * x2 * x2 * dry + 6 * c105dr9 * x2 * dry - 3 * c15dr7 * dry;
    T t4x1z = c945dr11 * x2 * x2 * drz + 6 * c105dr9 * x2 * drz - 3 * c15dr7 * drz;
    T t4y1x = c945dr11 * y2 * y2 * drx + 6 * c105dr9 * y2 * drx - 3 * c15dr7 * drx;
    T t4z1x = c945dr11 * z2 * z2 * drx + 6 * c105dr9 * z2 * drx - 3 * c15dr7 * drx;
    T t4y1z = c945dr11 * y2 * y2 * drz + 6 * c105dr9 * y2 * drz - 3 * c15dr7 * drz;
    T t4z1y = c945dr11 * z2 * z2 * dry + 6 * c105dr9 * z2 * dry - 3 * c15dr7 * dry;
    T t3x1y1z = c945dr11 * x2 * xyz + 3 * c105dr9 * xyz;
    T t3y1x1z = c945dr11 * y2 * xyz + 3 * c105dr9 * xyz;
    T t3z1x1y = c945dr11 * z2 * xyz + 3 * c105dr9 * xyz;
    T t3x2y = c945dr11 * x2 * drx * y2 + c105dr9 * drx * (3 * y2 + x2) - 3 * c15dr7 * drx;
    T t3y2x = c945dr11 * y2 * dry * x2 + c105dr9 * dry * (3 * x2 + y2) - 3 * c15dr7 * dry;
    T t3x2z = c945dr11 * x2 * drx * z2 + c105dr9 * drx * (3 * z2 + x2) - 3 * c15dr7 * drx;
    T t3z2x = c945dr11 * z2 * drz * x2 + c105dr9 * drz * (3 * x2 + z2) - 3 * c15dr7 * drz;
    T t3y2z = c945dr11 * y2 * dry * z2 + c105dr9 * dry * (3 * z2 + y2) - 3 * c15dr7 * dry;
    T t3z2y = c945dr11 * z2 * drz * y2 + c105dr9 * drz * (3 * y2 + z2) - 3 * c15dr7 * drz;
    T t2x2y1z = c945dr11 * xy * xyz + c105dr9 * drz * (x2 + y2) - c15dr7 * drz;
    T t2x2z1y = c945dr11 * xz * xyz + c105dr9 * dry * (x2 + z2) - c15dr7 * dry;
    T t2y2z1x = c945dr11 * yz * xyz + c105dr9 * drx * (y2 + z2) - c15dr7 * drx;

    *drx_g = c0prod * tx + tx_g * txx + ty_g * txy + tz_g * txz
            + txxx * txx_g + txxy * txy_g + txxz * txz_g + tyyx * tyy_g + txyz * tyz_g + tzzx * tzz_g
            + txxxx * txxx_g + txxxy * txxy_g + txxxz * txxz_g + tyyyx * tyyy_g + txxyy * tyyx_g + tyyxz * tyyz_g + tzzzx * tzzz_g + txxzz * tzzx_g + tzzxy * tzzy_g + txxyz * txyz_g
            + t5x * txxxx_g
            + t4x1y * txxxy_g
            + t4x1z * txxxz_g
            + t3x2y * txxyy_g
            + t3x2z * txxzz_g
            + t3x1y1z * txxyz_g
            + t4y1x * tyyyy_g
            + t3y2x * tyyyx_g
            + t3y1x1z * tyyyz_g
            + t2y2z1x * tyyzz_g
            + t2x2y1z * tyyxz_g
            + t4z1x * tzzzz_g
            + t3z2x * tzzzx_g
            + t3z1x1y * tzzzy_g
            + t2x2z1y * tzzxy_g; 

    *dry_g = c0prod * ty + tx_g * txy + ty_g * tyy + tz_g * tyz
            + txxy * txx_g + tyyx * txy_g + txyz * txz_g + tyyy * tyy_g + tyyz * tyz_g + tzzy * tzz_g
            + txxxy * txxx_g + txxyy * txxy_g + txxyz * txxz_g + tyyyy * tyyy_g + tyyyx * tyyx_g + tyyyz * tyyz_g + tzzzy * tzzz_g + tzzxy * tzzx_g + tyyzz * tzzy_g + tyyxz * txyz_g
            + t4x1y * txxxx_g
            + t3x2y * txxxy_g
            + t3x1y1z * txxxz_g
            + t3y2x * txxyy_g
            + t2x2z1y * txxzz_g
            + t2x2y1z * txxyz_g
            + t5y * tyyyy_g
            + t4y1x * tyyyx_g
            + t4y1z * tyyyz_g
            + t3y2z * tyyzz_g
            + t3y1x1z * tyyxz_g
            + t4z1y * tzzzz_g
            + t3z1x1y * tzzzx_g
            + t3z2y * tzzzy_g
            + t2y2z1x * tzzxy_g; 
    
    *drz_g = c0prod * tz + tx_g * txz + ty_g * tyz + tz_g * tzz
            + txxz * txx_g + txyz * txy_g + tzzx * txz_g + tyyz * tyy_g + tzzy * tyz_g + tzzz * tzz_g
            + txxxz * txxx_g + txxyz * txxy_g + txxzz * txxz_g + tyyyz * tyyy_g + tyyxz * tyyx_g + tyyzz * tyyz_g + tzzzz * tzzz_g + tzzzx * tzzx_g + tzzzy * tzzy_g + tzzxy * txyz_g
            + t4x1z * txxxx_g
            + t3x1y1z * txxxy_g
            + t3x2z * txxxz_g
            + t2x2y1z * txxyy_g
            + t3z2x * txxzz_g
            + t2x2z1y * txxyz_g
            + t4y1z * tyyyy_g
            + t3y1x1z * tyyyx_g
            + t3y2z * tyyyz_g
            + t3z2y * tyyzz_g
            + t2y2z1x * tyyxz_g
            + t5z * tzzzz_g
            + t4z1x * tzzzx_g
            + t4z1y * tzzzy_g
            + t3z1x1y * tzzxy_g;
}

#endif
