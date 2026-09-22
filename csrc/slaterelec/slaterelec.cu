// Slater-penetrated multipole electrostatics (rsfff.ff.electrostatics) as CUDA ops.
//
//   slater_elec_pair_energy  (coords, pairs, b, gate, m, n)         -> e (P,)
//   slater_elec_pair_grad    (coords, pairs, b, gate, m, n, g)      -> (d_coords, d_b, d_gate, d_m, d_n)
//   slater_elec_field        (coords, pairs, b, gate, m, n)         -> f (N, 10) = d(sum e)/dm
//   slater_elec_field_vjp    (coords, pairs, b, gate, m, n, lam)    -> (d_coords, d_b, d_gate, d_m, d_n)
//
// m, n are (N, K) polytensors with K in {1, 4, 10}; the kernels zero-pad to 10 and write
// multipole gradients as (N, 10) -- the caller slices. Nothing is materialised per pair: the
// damped tensors are rebuilt on the fly from (dr, b_i, b_j), which is what makes the field
// op a memory-light matvec for the coupled solve.
//
// The two VJP ops are the per-pair math (slater_elec.cuh) instantiated on Dual scalars with
// the cotangent as tangent seed: grad = (d/dm)[g * E], field VJP = (d/d*)[lam . dE/dm].
// Autograd wiring is in torchff/slaterelec.py.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include "common/dual.cuh"
#include "slaterelec/slater_elec.cuh"

namespace {

constexpr int BLOCK = 128;

inline int grid_for(int64_t n) {
    int64_t blocks = (n + BLOCK - 1) / BLOCK;
    return static_cast<int>(blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535);
}

template <typename S>
__device__ __forceinline__ void load_mp(const S* __restrict__ src, int64_t atom, int K, S* out) {
    for (int k = 0; k < 10; ++k) out[k] = k < K ? src[atom * K + k] : S(0);
}

template <typename S>
__device__ __forceinline__ void add_mp(S* __restrict__ dst, int64_t atom, const S* v, S w) {
    for (int k = 0; k < 10; ++k) atomicAdd(dst + atom * 10 + k, w * v[k]);
}

// ------------------------------------------------------------------------------------
// energy
// ------------------------------------------------------------------------------------
template <typename S>
__global__ void energy_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs, const S* __restrict__ b,
    const S* __restrict__ gate, const S* __restrict__ m, const S* __restrict__ n, int K, int64_t P,
    S* __restrict__ e_out
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < P; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S m_i[10], m_j[10], n_i[10], n_j[10];
        load_mp(m, i, K, m_i); load_mp(m, j, K, m_j); load_mp(n, i, K, n_i); load_mp(n, j, K, n_j);
        S dx = coords[3 * j] - coords[3 * i], dy = coords[3 * j + 1] - coords[3 * i + 1], dz = coords[3 * j + 2] - coords[3 * i + 2];
        S e, gmi[10], gmj[10], gni[10], gnj[10], gdr[3], gbi, gbj;
        slater_elec_pair_full<S>(m_i, m_j, n_i, n_j, dx, dy, dz, b[i], b[j], e, gmi, gmj, gni, gnj, gdr, gbi, gbj);
        e_out[p] = gate[p] * e;
    }
}

// ------------------------------------------------------------------------------------
// energy VJP:  d/d(coords, b, gate, m, n) of sum_p g_p gate_p E_p
// ------------------------------------------------------------------------------------
template <typename S>
__global__ void energy_grad_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs, const S* __restrict__ b,
    const S* __restrict__ gate, const S* __restrict__ m, const S* __restrict__ n, int K, int64_t P,
    const S* __restrict__ g,
    S* __restrict__ d_coords, S* __restrict__ d_b, S* __restrict__ d_gate, S* __restrict__ d_m, S* __restrict__ d_n
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < P; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S m_i[10], m_j[10], n_i[10], n_j[10];
        load_mp(m, i, K, m_i); load_mp(m, j, K, m_j); load_mp(n, i, K, n_i); load_mp(n, j, K, n_j);
        S dx = coords[3 * j] - coords[3 * i], dy = coords[3 * j + 1] - coords[3 * i + 1], dz = coords[3 * j + 2] - coords[3 * i + 2];
        S e, gmi[10], gmj[10], gni[10], gnj[10], gdr[3], gbi, gbj;
        slater_elec_pair_full<S>(m_i, m_j, n_i, n_j, dx, dy, dz, b[i], b[j], e, gmi, gmj, gni, gnj, gdr, gbi, gbj);
        S w = g[p] * gate[p];
        d_gate[p] = g[p] * e;
        atomicAdd(d_coords + 3 * i, -w * gdr[0]); atomicAdd(d_coords + 3 * i + 1, -w * gdr[1]); atomicAdd(d_coords + 3 * i + 2, -w * gdr[2]);
        atomicAdd(d_coords + 3 * j,  w * gdr[0]); atomicAdd(d_coords + 3 * j + 1,  w * gdr[1]); atomicAdd(d_coords + 3 * j + 2,  w * gdr[2]);
        atomicAdd(d_b + i, w * gbi); atomicAdd(d_b + j, w * gbj);
        add_mp(d_m, i, gmi, w); add_mp(d_m, j, gmj, w);
        add_mp(d_n, i, gni, w); add_mp(d_n, j, gnj, w);
    }
}

// ------------------------------------------------------------------------------------
// field (the coupled-solve matvec):  f_i = sum_p gate_p dE_p/dm_i
// ------------------------------------------------------------------------------------
template <typename S>
__global__ void field_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs, const S* __restrict__ b,
    const S* __restrict__ gate, const S* __restrict__ m, const S* __restrict__ n, int K, int64_t P,
    S* __restrict__ f
) {
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < P; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S m_i[10], m_j[10], n_i[10], n_j[10];
        load_mp(m, i, K, m_i); load_mp(m, j, K, m_j); load_mp(n, i, K, n_i); load_mp(n, j, K, n_j);
        S dx = coords[3 * j] - coords[3 * i], dy = coords[3 * j + 1] - coords[3 * i + 1], dz = coords[3 * j + 2] - coords[3 * i + 2];
        S gmi[10], gmj[10];
        slater_elec_pair_field<S>(m_i, m_j, n_i, n_j, dx, dy, dz, b[i], b[j], gmi, gmj);
        add_mp(f, i, gmi, gate[p]); add_mp(f, j, gmj, gate[p]);
    }
}

// ------------------------------------------------------------------------------------
// field VJP:  d/d(coords, b, gate, m, n) of sum_i lam_i . f_i   (lam seeds the m tangents)
// ------------------------------------------------------------------------------------
template <typename S>
__global__ void field_vjp_kernel(
    const S* __restrict__ coords, const int64_t* __restrict__ pairs, const S* __restrict__ b,
    const S* __restrict__ gate, const S* __restrict__ m, const S* __restrict__ n, int K, int64_t P,
    const S* __restrict__ lam,
    S* __restrict__ d_coords, S* __restrict__ d_b, S* __restrict__ d_gate, S* __restrict__ d_m, S* __restrict__ d_n
) {
    using D = Dual<S>;
    for (int64_t p = blockIdx.x * (int64_t)BLOCK + threadIdx.x; p < P; p += (int64_t)gridDim.x * BLOCK) {
        int64_t i = pairs[2 * p], j = pairs[2 * p + 1];
        S m_i[10], m_j[10], n_i[10], n_j[10], l_i[10], l_j[10];
        load_mp(m, i, K, m_i); load_mp(m, j, K, m_j); load_mp(n, i, K, n_i); load_mp(n, j, K, n_j);
        load_mp(lam, i, K, l_i); load_mp(lam, j, K, l_j);
        D mi[10], mj[10], ni[10], nj[10];
        for (int k = 0; k < 10; ++k) { mi[k] = D(m_i[k], l_i[k]); mj[k] = D(m_j[k], l_j[k]); ni[k] = D(n_i[k]); nj[k] = D(n_j[k]); }
        D dx(coords[3 * j] - coords[3 * i]), dy(coords[3 * j + 1] - coords[3 * i + 1]), dz(coords[3 * j + 2] - coords[3 * i + 2]);
        D e, gmi[10], gmj[10], gni[10], gnj[10], gdr[3], gbi, gbj;
        slater_elec_pair_full<D>(mi, mj, ni, nj, dx, dy, dz, D(b[i]), D(b[j]), e, gmi, gmj, gni, gnj, gdr, gbi, gbj);
        // tangent(e) = lam . dE/dm  -> d/dgate ; tangents of the gradients = d/d* of (lam . dE/dm)
        S w = gate[p];
        d_gate[p] = e.d;
        atomicAdd(d_coords + 3 * i, -w * gdr[0].d); atomicAdd(d_coords + 3 * i + 1, -w * gdr[1].d); atomicAdd(d_coords + 3 * i + 2, -w * gdr[2].d);
        atomicAdd(d_coords + 3 * j,  w * gdr[0].d); atomicAdd(d_coords + 3 * j + 1,  w * gdr[1].d); atomicAdd(d_coords + 3 * j + 2,  w * gdr[2].d);
        atomicAdd(d_b + i, w * gbi.d); atomicAdd(d_b + j, w * gbj.d);
        S t[10];
        for (int k = 0; k < 10; ++k) t[k] = gmi[k].d; add_mp(d_m, i, t, w);
        for (int k = 0; k < 10; ++k) t[k] = gmj[k].d; add_mp(d_m, j, t, w);
        for (int k = 0; k < 10; ++k) t[k] = gni[k].d; add_mp(d_n, i, t, w);
        for (int k = 0; k < 10; ++k) t[k] = gnj[k].d; add_mp(d_n, j, t, w);
    }
}

// ------------------------------------------------------------------------------------
// host wrappers
// ------------------------------------------------------------------------------------
#define CHECK_MP(m) TORCH_CHECK((m).dim() == 2 && ((m).size(1) == 1 || (m).size(1) == 4 || (m).size(1) == 10), #m " must be (N, 1|4|10)")

at::Tensor slater_elec_pair_energy_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& b, const at::Tensor& gate,
    const at::Tensor& m, const at::Tensor& n
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    CHECK_MP(m); CHECK_MP(n);
    int64_t P = pairs.size(0);
    int K = (int)m.size(1);
    auto e = at::empty({P}, coords.options());
    if (P == 0) return e;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "slater_elec_pair_energy", [&] {
        energy_kernel<scalar_t><<<grid_for(P), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(), b.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(), n.data_ptr<scalar_t>(), K, P, e.data_ptr<scalar_t>());
    });
    return e;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> slater_elec_pair_grad_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& b, const at::Tensor& gate,
    const at::Tensor& m, const at::Tensor& n, const at::Tensor& g
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    CHECK_MP(m); CHECK_MP(n);
    int64_t P = pairs.size(0);
    int K = (int)m.size(1);
    auto d_coords = at::zeros_like(coords);
    auto d_b = at::zeros_like(b);
    auto d_gate = at::empty_like(gate);
    auto d_m = at::zeros({m.size(0), 10}, m.options());
    auto d_n = at::zeros({n.size(0), 10}, n.options());
    if (P == 0) return {d_coords, d_b, d_gate, d_m, d_n};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "slater_elec_pair_grad", [&] {
        energy_grad_kernel<scalar_t><<<grid_for(P), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(), b.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(), n.data_ptr<scalar_t>(), K, P, g.data_ptr<scalar_t>(),
            d_coords.data_ptr<scalar_t>(), d_b.data_ptr<scalar_t>(), d_gate.data_ptr<scalar_t>(),
            d_m.data_ptr<scalar_t>(), d_n.data_ptr<scalar_t>());
    });
    return {d_coords, d_b, d_gate, d_m, d_n};
}

at::Tensor slater_elec_field_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& b, const at::Tensor& gate,
    const at::Tensor& m, const at::Tensor& n
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    CHECK_MP(m); CHECK_MP(n);
    int64_t P = pairs.size(0);
    int K = (int)m.size(1);
    auto f = at::zeros({m.size(0), 10}, m.options());
    if (P == 0) return f;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "slater_elec_field", [&] {
        field_kernel<scalar_t><<<grid_for(P), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(), b.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(), n.data_ptr<scalar_t>(), K, P, f.data_ptr<scalar_t>());
    });
    return f;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> slater_elec_field_vjp_cuda(
    const at::Tensor& coords, const at::Tensor& pairs, const at::Tensor& b, const at::Tensor& gate,
    const at::Tensor& m, const at::Tensor& n, const at::Tensor& lam
) {
    const c10::cuda::CUDAGuard guard(coords.device());
    CHECK_MP(m); CHECK_MP(n); CHECK_MP(lam);
    TORCH_CHECK(lam.size(1) == m.size(1), "lam must have the same K as m");
    int64_t P = pairs.size(0);
    int K = (int)m.size(1);
    auto d_coords = at::zeros_like(coords);
    auto d_b = at::zeros_like(b);
    auto d_gate = at::empty_like(gate);
    auto d_m = at::zeros({m.size(0), 10}, m.options());
    auto d_n = at::zeros({n.size(0), 10}, n.options());
    if (P == 0) return {d_coords, d_b, d_gate, d_m, d_n};
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(coords.scalar_type(), "slater_elec_field_vjp", [&] {
        field_vjp_kernel<scalar_t><<<grid_for(P), BLOCK, 0, stream>>>(
            coords.data_ptr<scalar_t>(), pairs.data_ptr<int64_t>(), b.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(),
            m.data_ptr<scalar_t>(), n.data_ptr<scalar_t>(), K, P, lam.data_ptr<scalar_t>(),
            d_coords.data_ptr<scalar_t>(), d_b.data_ptr<scalar_t>(), d_gate.data_ptr<scalar_t>(),
            d_m.data_ptr<scalar_t>(), d_n.data_ptr<scalar_t>());
    });
    return {d_coords, d_b, d_gate, d_m, d_n};
}

}  // namespace

TORCH_LIBRARY_IMPL(torchff, CUDA, m) {
    m.impl("slater_elec_pair_energy", slater_elec_pair_energy_cuda);
    m.impl("slater_elec_pair_grad", slater_elec_pair_grad_cuda);
    m.impl("slater_elec_field", slater_elec_field_cuda);
    m.impl("slater_elec_field_vjp", slater_elec_field_vjp_cuda);
}
