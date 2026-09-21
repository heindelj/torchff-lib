// Per-element force-field terms with per-pair output and double backward.
//
// Three ops per term:  <term>_energy  -> (P,) energies
//                      <term>_grad    -> (dE/dcoords, dE/dparam...) scaled by the incoming g (P,)
//                      <term>_hvp     -> the backward of _grad: given cotangents v for each of
//                                        its outputs, returns the gradients w.r.t. coords,
//                                        params and g.
// The _grad and _hvp kernels are one template: _hvp instantiates the per-term math on
// Dual<scalar_t> (common/dual.cuh) with the cotangents as tangent seeds, so the primal part
// reproduces the gradient and the tangent part is the Hessian-vector product.
//
// Autograd is wired in Python (torchff/ffterms.py); these ops are plain CUDA implementations.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "common/dual.cuh"
#include "ffterms/terms.cuh"

namespace {

constexpr int BLOCK = 128;

inline int grid_for(int64_t n) {
    int64_t blocks = (n + BLOCK - 1) / BLOCK;
    return static_cast<int>(blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535);
}

template <typename S>
__device__ __forceinline__ void scatter3(S* out, int64_t atom, S x, S y, S z) {
    atomicAdd(out + 3 * atom,     x);
    atomicAdd(out + 3 * atom + 1, y);
    atomicAdd(out + 3 * atom + 2, z);
}

// =====================================================================================
// Tang-Toennies dispersion
// =====================================================================================

template <typename S>
__global__ void tt_energy_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs,
    const S* __restrict__ c6, const S* __restrict__ b, int64_t n, S* __restrict__ e_out
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S dx = coords[3 * j] - coords[3 * i];
        S dy = coords[3 * j + 1] - coords[3 * i + 1];
        S dz = coords[3 * j + 2] - coords[3 * i + 2];
        S e, grad[5];
        tt_dispersion_term<S, S>(dx, dy, dz, c6[p], b[p], e, grad);
        e_out[p] = e;
    }
}

template <typename S, bool HVP>
__global__ void tt_grad_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs,
    const S* __restrict__ c6, const S* __restrict__ b, const S* __restrict__ g, int64_t n,
    const S* __restrict__ v_coords, const S* __restrict__ v_c6, const S* __restrict__ v_b,
    S* __restrict__ o_coords, S* __restrict__ o_c6, S* __restrict__ o_b, S* __restrict__ o_g
) {
    using T = typename std::conditional<HVP, Dual<S>, S>::type;
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S dx = coords[3 * j] - coords[3 * i];
        S dy = coords[3 * j + 1] - coords[3 * i + 1];
        S dz = coords[3 * j + 2] - coords[3 * i + 2];
        T e, grad[5];
        if constexpr (HVP) {
            S tdx = v_coords[3 * j] - v_coords[3 * i];
            S tdy = v_coords[3 * j + 1] - v_coords[3 * i + 1];
            S tdz = v_coords[3 * j + 2] - v_coords[3 * i + 2];
            tt_dispersion_term<T, S>(T(dx, tdx), T(dy, tdy), T(dz, tdz), T(c6[p], v_c6[p]), T(b[p], v_b[p]), e, grad);
            // d/dg: the JVP  grad . v
            o_g[p] = grad[0].v * tdx + grad[1].v * tdy + grad[2].v * tdz + grad[3].v * v_c6[p] + grad[4].v * v_b[p];
            S gp = g[p];
            scatter3(o_coords, i, -gp * grad[0].d, -gp * grad[1].d, -gp * grad[2].d);
            scatter3(o_coords, j,  gp * grad[0].d,  gp * grad[1].d,  gp * grad[2].d);
            o_c6[p] = gp * grad[3].d;
            o_b[p] = gp * grad[4].d;
        } else {
            tt_dispersion_term<S, S>(dx, dy, dz, c6[p], b[p], e, grad);
            S gp = g[p];
            scatter3(o_coords, i, -gp * grad[0], -gp * grad[1], -gp * grad[2]);
            scatter3(o_coords, j,  gp * grad[0],  gp * grad[1],  gp * grad[2]);
            o_c6[p] = gp * grad[3];
            o_b[p] = gp * grad[4];
        }
    }
}

at::Tensor tt_dispersion_pair_energy_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& c6, const at::Tensor& b
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = pairs.size(0);
    auto e = at::empty({n}, coords.options());
    if (n == 0) return e;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "tt_dispersion_pair_energy", [&] {
        tt_energy_kernel<scalar_t><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(),
            c6.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), n, e.data_ptr<scalar_t>());
    });
    return e;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> tt_dispersion_pair_grad_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& c6, const at::Tensor& b,
    const at::Tensor& g
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = pairs.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_c6 = at::empty_like(c6);
    auto o_b = at::empty_like(b);
    if (n == 0) return {o_coords, o_c6, o_b};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "tt_dispersion_pair_grad", [&] {
        tt_grad_kernel<scalar_t, false><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(),
            c6.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), n,
            nullptr, nullptr, nullptr,
            o_coords.data_ptr<scalar_t>(), o_c6.data_ptr<scalar_t>(), o_b.data_ptr<scalar_t>(), nullptr);
    });
    return {o_coords, o_c6, o_b};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tt_dispersion_pair_hvp_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& c6, const at::Tensor& b,
    const at::Tensor& g, const at::Tensor& v_coords, const at::Tensor& v_c6, const at::Tensor& v_b
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = pairs.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_c6 = at::empty_like(c6);
    auto o_b = at::empty_like(b);
    auto o_g = at::empty_like(g);
    if (n == 0) return {o_coords, o_c6, o_b, o_g};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "tt_dispersion_pair_hvp", [&] {
        tt_grad_kernel<scalar_t, true><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(),
            c6.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), n,
            v_coords.data_ptr<scalar_t>(), v_c6.data_ptr<scalar_t>(), v_b.data_ptr<scalar_t>(),
            o_coords.data_ptr<scalar_t>(), o_c6.data_ptr<scalar_t>(), o_b.data_ptr<scalar_t>(), o_g.data_ptr<scalar_t>());
    });
    return {o_coords, o_c6, o_b, o_g};
}

// =====================================================================================
// Morse bond
// =====================================================================================

template <typename S>
__global__ void morse_energy_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ bonds,
    const S* __restrict__ r_eq, const S* __restrict__ dd, const S* __restrict__ k, int64_t n,
    S* __restrict__ e_out
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = bonds[2 * p], j = bonds[2 * p + 1];
        S dx = coords[3 * j] - coords[3 * i];
        S dy = coords[3 * j + 1] - coords[3 * i + 1];
        S dz = coords[3 * j + 2] - coords[3 * i + 2];
        S e, grad[6];
        morse_term<S, S>(dx, dy, dz, r_eq[p], dd[p], k[p], e, grad);
        e_out[p] = e;
    }
}

template <typename S, bool HVP>
__global__ void morse_grad_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ bonds,
    const S* __restrict__ r_eq, const S* __restrict__ dd, const S* __restrict__ k,
    const S* __restrict__ g, int64_t n,
    const S* __restrict__ v_coords, const S* __restrict__ v_req, const S* __restrict__ v_d, const S* __restrict__ v_k,
    S* __restrict__ o_coords, S* __restrict__ o_req, S* __restrict__ o_d, S* __restrict__ o_k, S* __restrict__ o_g
) {
    using T = typename std::conditional<HVP, Dual<S>, S>::type;
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = bonds[2 * p], j = bonds[2 * p + 1];
        S dx = coords[3 * j] - coords[3 * i];
        S dy = coords[3 * j + 1] - coords[3 * i + 1];
        S dz = coords[3 * j + 2] - coords[3 * i + 2];
        T e, grad[6];
        S gp = g[p];
        if constexpr (HVP) {
            S tdx = v_coords[3 * j] - v_coords[3 * i];
            S tdy = v_coords[3 * j + 1] - v_coords[3 * i + 1];
            S tdz = v_coords[3 * j + 2] - v_coords[3 * i + 2];
            morse_term<T, S>(T(dx, tdx), T(dy, tdy), T(dz, tdz),
                             T(r_eq[p], v_req[p]), T(dd[p], v_d[p]), T(k[p], v_k[p]), e, grad);
            o_g[p] = grad[0].v * tdx + grad[1].v * tdy + grad[2].v * tdz
                   + grad[3].v * v_req[p] + grad[4].v * v_d[p] + grad[5].v * v_k[p];
            scatter3(o_coords, i, -gp * grad[0].d, -gp * grad[1].d, -gp * grad[2].d);
            scatter3(o_coords, j,  gp * grad[0].d,  gp * grad[1].d,  gp * grad[2].d);
            o_req[p] = gp * grad[3].d;
            o_d[p] = gp * grad[4].d;
            o_k[p] = gp * grad[5].d;
        } else {
            morse_term<S, S>(dx, dy, dz, r_eq[p], dd[p], k[p], e, grad);
            scatter3(o_coords, i, -gp * grad[0], -gp * grad[1], -gp * grad[2]);
            scatter3(o_coords, j,  gp * grad[0],  gp * grad[1],  gp * grad[2]);
            o_req[p] = gp * grad[3];
            o_d[p] = gp * grad[4];
            o_k[p] = gp * grad[5];
        }
    }
}

at::Tensor morse_bond_energy_cuda(
    const at::Tensor& coords, const at::Tensor& bonds,
    const at::Tensor& r_eq, const at::Tensor& d, const at::Tensor& k
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = bonds.size(0);
    auto e = at::empty({n}, coords.options());
    if (n == 0) return e;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "morse_bond_energy", [&] {
        morse_energy_kernel<scalar_t><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), bonds.data_ptr<int64_t>(),
            r_eq.data_ptr<scalar_t>(), d.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), n,
            e.data_ptr<scalar_t>());
    });
    return e;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> morse_bond_grad_cuda(
    const at::Tensor& coords, const at::Tensor& bonds,
    const at::Tensor& r_eq, const at::Tensor& d, const at::Tensor& k, const at::Tensor& g
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = bonds.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_req = at::empty_like(r_eq);
    auto o_d = at::empty_like(d);
    auto o_k = at::empty_like(k);
    if (n == 0) return {o_coords, o_req, o_d, o_k};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "morse_bond_grad", [&] {
        morse_grad_kernel<scalar_t, false><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), bonds.data_ptr<int64_t>(),
            r_eq.data_ptr<scalar_t>(), d.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(), n, nullptr, nullptr, nullptr, nullptr,
            o_coords.data_ptr<scalar_t>(), o_req.data_ptr<scalar_t>(), o_d.data_ptr<scalar_t>(),
            o_k.data_ptr<scalar_t>(), nullptr);
    });
    return {o_coords, o_req, o_d, o_k};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> morse_bond_hvp_cuda(
    const at::Tensor& coords, const at::Tensor& bonds,
    const at::Tensor& r_eq, const at::Tensor& d, const at::Tensor& k, const at::Tensor& g,
    const at::Tensor& v_coords, const at::Tensor& v_req, const at::Tensor& v_d, const at::Tensor& v_k
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = bonds.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_req = at::empty_like(r_eq);
    auto o_d = at::empty_like(d);
    auto o_k = at::empty_like(k);
    auto o_g = at::empty_like(g);
    if (n == 0) return {o_coords, o_req, o_d, o_k, o_g};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "morse_bond_hvp", [&] {
        morse_grad_kernel<scalar_t, true><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), bonds.data_ptr<int64_t>(),
            r_eq.data_ptr<scalar_t>(), d.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            g.data_ptr<scalar_t>(), n,
            v_coords.data_ptr<scalar_t>(), v_req.data_ptr<scalar_t>(), v_d.data_ptr<scalar_t>(), v_k.data_ptr<scalar_t>(),
            o_coords.data_ptr<scalar_t>(), o_req.data_ptr<scalar_t>(), o_d.data_ptr<scalar_t>(),
            o_k.data_ptr<scalar_t>(), o_g.data_ptr<scalar_t>());
    });
    return {o_coords, o_req, o_d, o_k, o_g};
}

// =====================================================================================
// Cosine-harmonic angle
// =====================================================================================

template <typename S>
__global__ void angle_energy_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ angles,
    const S* __restrict__ cos_eq, const S* __restrict__ k, int64_t n, S* __restrict__ e_out
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t a = angles[3 * p], apex = angles[3 * p + 1], c = angles[3 * p + 2];
        S v1x = coords[3 * a] - coords[3 * apex], v1y = coords[3 * a + 1] - coords[3 * apex + 1], v1z = coords[3 * a + 2] - coords[3 * apex + 2];
        S v2x = coords[3 * c] - coords[3 * apex], v2y = coords[3 * c + 1] - coords[3 * apex + 1], v2z = coords[3 * c + 2] - coords[3 * apex + 2];
        S e, grad[8];
        cosine_angle_term<S, S>(v1x, v1y, v1z, v2x, v2y, v2z, cos_eq[p], k[p], e, grad);
        e_out[p] = e;
    }
}

template <typename S, bool HVP>
__global__ void angle_grad_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ angles,
    const S* __restrict__ cos_eq, const S* __restrict__ k, const S* __restrict__ g, int64_t n,
    const S* __restrict__ v_coords, const S* __restrict__ v_cos, const S* __restrict__ v_k,
    S* __restrict__ o_coords, S* __restrict__ o_cos, S* __restrict__ o_k, S* __restrict__ o_g
) {
    using T = typename std::conditional<HVP, Dual<S>, S>::type;
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < n; p += (int64_t)gridDim.x * BLOCK) {
        int64_t a = angles[3 * p], apex = angles[3 * p + 1], c = angles[3 * p + 2];
        S v1[3], v2[3];
        for (int q = 0; q < 3; ++q) {
            v1[q] = coords[3 * a + q] - coords[3 * apex + q];
            v2[q] = coords[3 * c + q] - coords[3 * apex + q];
        }
        T e, grad[8];
        S gp = g[p];
        if constexpr (HVP) {
            S t1[3], t2[3];
            for (int q = 0; q < 3; ++q) {
                t1[q] = v_coords[3 * a + q] - v_coords[3 * apex + q];
                t2[q] = v_coords[3 * c + q] - v_coords[3 * apex + q];
            }
            cosine_angle_term<T, S>(T(v1[0], t1[0]), T(v1[1], t1[1]), T(v1[2], t1[2]),
                                    T(v2[0], t2[0]), T(v2[1], t2[1]), T(v2[2], t2[2]),
                                    T(cos_eq[p], v_cos[p]), T(k[p], v_k[p]), e, grad);
            o_g[p] = grad[0].v * t1[0] + grad[1].v * t1[1] + grad[2].v * t1[2]
                   + grad[3].v * t2[0] + grad[4].v * t2[1] + grad[5].v * t2[2]
                   + grad[6].v * v_cos[p] + grad[7].v * v_k[p];
            S g1x = gp * grad[0].d, g1y = gp * grad[1].d, g1z = gp * grad[2].d;
            S g2x = gp * grad[3].d, g2y = gp * grad[4].d, g2z = gp * grad[5].d;
            scatter3(o_coords, a, g1x, g1y, g1z);
            scatter3(o_coords, c, g2x, g2y, g2z);
            scatter3(o_coords, apex, -(g1x + g2x), -(g1y + g2y), -(g1z + g2z));
            o_cos[p] = gp * grad[6].d;
            o_k[p] = gp * grad[7].d;
        } else {
            cosine_angle_term<S, S>(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], cos_eq[p], k[p], e, grad);
            S g1x = gp * grad[0], g1y = gp * grad[1], g1z = gp * grad[2];
            S g2x = gp * grad[3], g2y = gp * grad[4], g2z = gp * grad[5];
            scatter3(o_coords, a, g1x, g1y, g1z);
            scatter3(o_coords, c, g2x, g2y, g2z);
            scatter3(o_coords, apex, -(g1x + g2x), -(g1y + g2y), -(g1z + g2z));
            o_cos[p] = gp * grad[6];
            o_k[p] = gp * grad[7];
        }
    }
}

at::Tensor cosine_angle_energy_cuda(
    const at::Tensor& coords, const at::Tensor& angles, const at::Tensor& cos_eq, const at::Tensor& k
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = angles.size(0);
    auto e = at::empty({n}, coords.options());
    if (n == 0) return e;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cosine_angle_energy", [&] {
        angle_energy_kernel<scalar_t><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), angles.data_ptr<int64_t>(),
            cos_eq.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), n, e.data_ptr<scalar_t>());
    });
    return e;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> cosine_angle_grad_cuda(
    const at::Tensor& coords, const at::Tensor& angles, const at::Tensor& cos_eq, const at::Tensor& k,
    const at::Tensor& g
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = angles.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_cos = at::empty_like(cos_eq);
    auto o_k = at::empty_like(k);
    if (n == 0) return {o_coords, o_cos, o_k};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cosine_angle_grad", [&] {
        angle_grad_kernel<scalar_t, false><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), angles.data_ptr<int64_t>(),
            cos_eq.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), n,
            nullptr, nullptr, nullptr,
            o_coords.data_ptr<scalar_t>(), o_cos.data_ptr<scalar_t>(), o_k.data_ptr<scalar_t>(), nullptr);
    });
    return {o_coords, o_cos, o_k};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cosine_angle_hvp_cuda(
    const at::Tensor& coords, const at::Tensor& angles, const at::Tensor& cos_eq, const at::Tensor& k,
    const at::Tensor& g, const at::Tensor& v_coords, const at::Tensor& v_cos, const at::Tensor& v_k
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    int64_t n = angles.size(0);
    auto o_coords = at::zeros_like(coords);
    auto o_cos = at::empty_like(cos_eq);
    auto o_k = at::empty_like(k);
    auto o_g = at::empty_like(g);
    if (n == 0) return {o_coords, o_cos, o_k, o_g};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "cosine_angle_hvp", [&] {
        angle_grad_kernel<scalar_t, true><<<grid_for(n), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), angles.data_ptr<int64_t>(),
            cos_eq.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), n,
            v_coords.data_ptr<scalar_t>(), v_cos.data_ptr<scalar_t>(), v_k.data_ptr<scalar_t>(),
            o_coords.data_ptr<scalar_t>(), o_cos.data_ptr<scalar_t>(), o_k.data_ptr<scalar_t>(), o_g.data_ptr<scalar_t>());
    });
    return {o_coords, o_cos, o_k, o_g};
}

}  // namespace

TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("tt_dispersion_pair_energy", tt_dispersion_pair_energy_cuda);
    m.impl("tt_dispersion_pair_grad", tt_dispersion_pair_grad_cuda);
    m.impl("tt_dispersion_pair_hvp", tt_dispersion_pair_hvp_cuda);
    m.impl("morse_bond_energy", morse_bond_energy_cuda);
    m.impl("morse_bond_grad", morse_bond_grad_cuda);
    m.impl("morse_bond_hvp", morse_bond_hvp_cuda);
    m.impl("cosine_angle_energy", cosine_angle_energy_cuda);
    m.impl("cosine_angle_grad", cosine_angle_grad_cuda);
    m.impl("cosine_angle_hvp", cosine_angle_hvp_cuda);
}
