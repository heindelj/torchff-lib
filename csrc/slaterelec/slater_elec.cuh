#ifndef TORCHFF_SLATER_ELEC_CUH
#define TORCHFF_SLATER_ELEC_CUH

// Slater-penetrated multipole electrostatics between two atoms (rsfff.ff.electrostatics):
//
//   E_ij = m_j^T T0 m_i                       point multipoles, undamped 1/r tail
//        + s_j^T Tss s_i                      shell-shell, two-center Slater overlap,  b_ij = sqrt(b_i b_j)
//        + n_j^T T1c(b_i) s_i                 nucleus j sees shell i, one-center
//        + s_j^T T1c(b_j) n_i                 shell j sees nucleus i, one-center
//
// with m the full multipoles, n the point nuclear charge (polytensor with only slot 0) and
// s = m - n the shell. The damped tensors carry a leading minus: the damping functions are the
// overlap complement (pyCMM convention), so Tss = -f_2c * T.
//
// Everything is templated on the scalar type T (plain or Dual, common/dual.cuh) and the
// polytensor is always the full rank-2 layout of 10 slots; lower ranks pass zeros, which
// gives the lower-rank energy exactly (the tensor is block triangular in rank).
//
// Units are the caller's; rsfff passes atomic units (bohr, e, Hartree).

#include "common/dual.cuh"
#include "slaterelec/multipole_pair.cuh"

// ---------------------------------------------------------------------------------------
// Slater damping polynomials (pyCMM; identical to rsfff.ff.multipole and csrc/cmm/damps.cuh).
// d[0..5] scale 1/r, 1/r^3, ..., 1/r^11. The one-center family builds its 1/r^9 factor on
// the 1/r^5 polynomial, not on 1/r^7 -- transcribed, not "corrected", because that is the
// fitted model's functional form and it is a consistent derivative family as written.
// ---------------------------------------------------------------------------------------
template <typename T, typename S>
__device__ __forceinline__ void one_center_damps(T u, T* d) {
    T u2 = u * u;
    T u4 = u2 * u2;
    T expu = d_exp(-u);
    T p = S(1) + u * S(0.5);
    d[0] = expu * p;
    p = S(1) + u + u2 * S(0.5);
    d[1] = expu * p;
    p = p + u2 * u * S(1.0 / 6.0);
    d[2] = expu * p;
    d[3] = expu * (p + u4 * S(1.0 / 30.0));
    d[4] = expu * (p + u4 * S(4.0 / 105.0) + u4 * u * S(1.0 / 210.0));
    d[5] = expu * (p + u4 * S(5.0 / 126.0) + u4 * u * S(2.0 / 315.0) + u2 * u4 * S(1.0 / 1890.0));
}

template <typename T, typename S>
__device__ __forceinline__ void two_center_damps(T u, T* d) {
    T u2 = u * u;
    T u4 = u2 * u2;
    T expu = d_exp(-u);
    T p = S(1) + u * S(11.0 / 16.0) + u2 * S(3.0 / 16.0) + u2 * u * S(1.0 / 48.0);
    d[0] = expu * p;
    p = S(1) + u + u2 * S(0.5);
    d[1] = expu * (p + u2 * u * S(7.0 / 48.0) + u4 * S(1.0 / 48.0));
    p = p + u2 * u * S(1.0 / 6.0) + u4 * S(1.0 / 24.0);
    d[2] = expu * (p + u4 * u * S(1.0 / 144.0));
    p = p + u4 * u * S(1.0 / 120.0) + u4 * u2 * S(1.0 / 720.0);
    d[3] = expu * p;
    p = p + u4 * u2 * u * S(1.0 / 5040.0);
    d[4] = expu * p;
    p = p + u4 * u4 * S(1.0 / 45360.0);
    d[5] = expu * p;
}

// ---------------------------------------------------------------------------------------
// One damped term  w * a_j^T T(dr; damp) a_i, accumulated into e and the gradients.
// g_i / g_j are d/da_i, d/da_j (the potential, field and field gradient at each site);
// gdr is d/d(dr). All three are *added to*, scaled by w.
// ---------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void mp_term(
    const T* a_i, const T* a_j, T drx, T dry, T drz, const T* d, T w,
    T& e, T* g_i, T* g_j, T* gdr
) {
    T ene, gi[10], gj[10], gx, gy, gz;
    mp_pair_grad<T>(
        a_i[0], a_i[1], a_i[2], a_i[3], a_i[4], a_i[5], a_i[6], a_i[7], a_i[8], a_i[9],
        a_j[0], a_j[1], a_j[2], a_j[3], a_j[4], a_j[5], a_j[6], a_j[7], a_j[8], a_j[9],
        drx, dry, drz, d[0], d[1], d[2], d[3], d[4], d[5],
        &ene,
        gi, gi + 1, gi + 2, gi + 3, gi + 4, gi + 5, gi + 6, gi + 7, gi + 8, gi + 9,
        gj, gj + 1, gj + 2, gj + 3, gj + 4, gj + 5, gj + 6, gj + 7, gj + 8, gj + 9,
        &gx, &gy, &gz);
    e += w * ene;
    for (int k = 0; k < 10; ++k) { g_i[k] += w * gi[k]; g_j[k] += w * gj[k]; }
    gdr[0] += w * gx; gdr[1] += w * gy; gdr[2] += w * gz;
}

// ---------------------------------------------------------------------------------------
// The full pair: energy and every first derivative. Inputs m_i, m_j (10), n_i, n_j (10, only
// slot 0 nonzero), dr = r_j - r_i, exponents b_i, b_j.
// Outputs: e; g_mi, g_mj = dE/dm; g_ni, g_nj = dE/dn; gdr = dE/d(dr); g_bi, g_bj = dE/db.
// The b derivatives come from a Dual evaluation of the three damped terms seeded with
// du/db: one extra pass per damped term rather than a second hand-written derivative.
// ---------------------------------------------------------------------------------------
template <typename S>
__device__ __forceinline__ void slater_elec_pair_full(
    const S* m_i, const S* m_j, const S* n_i, const S* n_j,
    S drx, S dry, S drz, S b_i, S b_j,
    S& e, S* g_mi, S* g_mj, S* g_ni, S* g_nj, S* gdr, S& g_bi, S& g_bj
) {
    using D = Dual<S>;
    S r = d_sqrt(drx * drx + dry * dry + drz * drz);
    S b_ij = d_sqrt(b_i * b_j);
    S s_i[10], s_j[10];
    for (int k = 0; k < 10; ++k) { s_i[k] = m_i[k] - n_i[k]; s_j[k] = m_j[k] - n_j[k]; }
    for (int k = 0; k < 10; ++k) { g_mi[k] = S(0); g_mj[k] = S(0); g_ni[k] = S(0); g_nj[k] = S(0); }
    gdr[0] = gdr[1] = gdr[2] = S(0);
    e = S(0);

    // 1. point term, undamped
    {
        S one[6] = {S(1), S(1), S(1), S(1), S(1), S(1)};
        mp_term<S>(m_i, m_j, drx, dry, drz, one, S(1), e, g_mi, g_mj, gdr);
    }

    // 2-4. damped terms on Dual<S>: tangent = d/du of the damps, so tangent(e) = dE/du.
    D dx(drx), dy(dry), dz(drz);
    D si[10], sj[10], ni[10], nj[10];
    for (int k = 0; k < 10; ++k) { si[k] = D(s_i[k]); sj[k] = D(s_j[k]); ni[k] = D(n_i[k]); nj[k] = D(n_j[k]); }

    auto damped = [&](const D* a_i, const D* a_j, D u, bool two_center,
                      S* out_i, S* out_j, S& de_du) {
        D d[6];
        if (two_center) two_center_damps<D, S>(u, d); else one_center_damps<D, S>(u, d);
        for (int k = 0; k < 6; ++k) d[k] = -d[k];           // overlap complement convention
        D ee(S(0)), gi[10], gj[10], gd[3];
        for (int k = 0; k < 10; ++k) { gi[k] = D(S(0)); gj[k] = D(S(0)); }
        gd[0] = gd[1] = gd[2] = D(S(0));
        mp_term<D>(a_i, a_j, dx, dy, dz, d, D(S(1)), ee, gi, gj, gd);
        e += ee.v;
        for (int k = 0; k < 10; ++k) { out_i[k] += gi[k].v; out_j[k] += gj[k].v; }
        gdr[0] += gd[0].v; gdr[1] += gd[1].v; gdr[2] += gd[2].v;
        de_du = ee.d;
    };

    S de_du_ss, de_du_1ci, de_du_1cj;
    S gsi[10] = {}, gsj[10] = {}, gni_[10] = {}, gnj_[10] = {};

    // shell-shell: u = b_ij r, tangent 1 -> dE/du
    damped(si, sj, D(b_ij * r, S(1)), true, gsi, gsj, de_du_ss);
    // nucleus j sees shell i: a_i = s_i, a_j = n_j, u = b_i r
    damped(si, nj, D(b_i * r, S(1)), false, gsi, gnj_, de_du_1ci);
    // shell j sees nucleus i: a_i = n_i, a_j = s_j, u = b_j r
    damped(ni, sj, D(b_j * r, S(1)), false, gni_, gsj, de_du_1cj);

    // s = m - n: chain the shell gradients onto m and n
    for (int k = 0; k < 10; ++k) {
        g_mi[k] += gsi[k];  g_ni[k] += gni_[k] - gsi[k];
        g_mj[k] += gsj[k];  g_nj[k] += gnj_[k] - gsj[k];
    }
    // b: u_ss = sqrt(b_i b_j) r, u_1ci = b_i r, u_1cj = b_j r
    g_bi = de_du_ss * r * b_ij / (S(2) * b_i) + de_du_1ci * r;
    g_bj = de_du_ss * r * b_ij / (S(2) * b_j) + de_du_1cj * r;
    // the damps also depend on r through u -- the Dual pass differentiated the damps w.r.t. u
    // but the generated dr gradient inside mp_pair_grad treats the damps via the derivative
    // family (next-order damping), which already accounts for d(damp)/dr. Nothing to add.
}

// ---------------------------------------------------------------------------------------
// The matvec-only variant: dE/dm at both sites (the potential / field / field gradient),
// no dr or b derivatives. This is what the coupled solve calls every CG iteration.
// ---------------------------------------------------------------------------------------
template <typename S>
__device__ __forceinline__ void slater_elec_pair_field(
    const S* m_i, const S* m_j, const S* n_i, const S* n_j,
    S drx, S dry, S drz, S b_i, S b_j,
    S* g_mi, S* g_mj
) {
    S r = d_sqrt(drx * drx + dry * dry + drz * drz);
    S b_ij = d_sqrt(b_i * b_j);
    S s_i[10], s_j[10];
    for (int k = 0; k < 10; ++k) { s_i[k] = m_i[k] - n_i[k]; s_j[k] = m_j[k] - n_j[k]; }
    for (int k = 0; k < 10; ++k) { g_mi[k] = S(0); g_mj[k] = S(0); }
    S e = S(0), gdr[3] = {S(0), S(0), S(0)};
    S junk_i[10], junk_j[10];
    for (int k = 0; k < 10; ++k) { junk_i[k] = S(0); junk_j[k] = S(0); }

    S one[6] = {S(1), S(1), S(1), S(1), S(1), S(1)};
    mp_term<S>(m_i, m_j, drx, dry, drz, one, S(1), e, g_mi, g_mj, gdr);

    S d[6];
    two_center_damps<S, S>(b_ij * r, d);
    for (int k = 0; k < 6; ++k) d[k] = -d[k];
    mp_term<S>(s_i, s_j, drx, dry, drz, d, S(1), e, g_mi, g_mj, gdr);       // ds_i = dm_i

    one_center_damps<S, S>(b_i * r, d);
    for (int k = 0; k < 6; ++k) d[k] = -d[k];
    mp_term<S>(s_i, n_j, drx, dry, drz, d, S(1), e, g_mi, junk_j, gdr);     // n_j side: no m grad

    one_center_damps<S, S>(b_j * r, d);
    for (int k = 0; k < 6; ++k) d[k] = -d[k];
    mp_term<S>(n_i, s_j, drx, dry, drz, d, S(1), e, junk_i, g_mj, gdr);
}

// ---------------------------------------------------------------------------------------
// Slater multipolar Pauli repulsion (rsfff.ff.pauli.slater_pauli_pair_energy):
//
//   E_ij = a_j^T [ f_2c(b_ij r) T ] a_i
//
// with a the *Pauli* polytensors (emitted directly, already gathered onto the pair) and one
// combined exponent b_ij per pair. Same damping family as the shell-shell term above, with
// the overlap complement's sign flipped: the Pauli energy is +f T, the penetration is -f T.
// Outputs: e; g_ai, g_aj = dE/da; gdr = dE/d(dr); g_b = dE/db_ij (from a Dual pass in u).
// ---------------------------------------------------------------------------------------
template <typename S>
__device__ __forceinline__ void slater_pauli_pair_full(
    const S* a_i, const S* a_j, S drx, S dry, S drz, S b_ij,
    S& e, S* g_ai, S* g_aj, S* gdr, S& g_b
) {
    using D = Dual<S>;
    S r = d_sqrt(drx * drx + dry * dry + drz * drz);
    D ai[10], aj[10];
    for (int k = 0; k < 10; ++k) { ai[k] = D(a_i[k]); aj[k] = D(a_j[k]); }
    D d[6];
    two_center_damps<D, S>(D(b_ij * r, S(1)), d);        // tangent = d/du
    D ee(S(0)), gi[10], gj[10], gd[3];
    for (int k = 0; k < 10; ++k) { gi[k] = D(S(0)); gj[k] = D(S(0)); }
    gd[0] = gd[1] = gd[2] = D(S(0));
    mp_term<D>(ai, aj, D(drx), D(dry), D(drz), d, D(S(1)), ee, gi, gj, gd);
    e = ee.v;
    for (int k = 0; k < 10; ++k) { g_ai[k] = gi[k].v; g_aj[k] = gj[k].v; }
    gdr[0] = gd[0].v; gdr[1] = gd[1].v; gdr[2] = gd[2].v;
    g_b = ee.d * r;                                      // u = b_ij r
}

#endif /* TORCHFF_SLATER_ELEC_CUH */
