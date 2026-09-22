// Host build of csrc/slaterelec/slater_elec.cuh: prints energy and every first derivative of
// the Slater-penetrated multipole pair for fixed inputs, and the matvec variant's fields.
// tests/test_slaterelec.py compares against rsfff-equivalent torch autograd.
#define __device__
#define __forceinline__ inline
#include <cmath>
#include <cstdio>
static inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
static inline double rsqrt(double x) { return 1.0 / sqrt(x); }
#include "common/dual.cuh"
#include "slaterelec/slater_elec.cuh"

int main() {
    double m_i[10] = {0.31, 0.12, -0.05, 0.08, 0.021, -0.013, 0.007, -0.011, 0.004, -0.010};
    double m_j[10] = {-0.42, -0.07, 0.11, 0.02, -0.015, 0.009, 0.012, 0.006, -0.008, 0.009};
    double n_i[10] = {1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double n_j[10] = {6.0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double dr[3] = {2.3, -1.1, 0.7};
    double b_i = 1.9, b_j = 2.4;
    double e, g_mi[10], g_mj[10], g_ni[10], g_nj[10], gdr[3], g_bi, g_bj;
    slater_elec_pair_full<double>(m_i, m_j, n_i, n_j, dr[0], dr[1], dr[2], b_i, b_j,
                                  e, g_mi, g_mj, g_ni, g_nj, gdr, g_bi, g_bj);
    printf("e %.17e\n", e);
    for (int k = 0; k < 10; ++k) printf("g_mi %d %.17e\n", k, g_mi[k]);
    for (int k = 0; k < 10; ++k) printf("g_mj %d %.17e\n", k, g_mj[k]);
    for (int k = 0; k < 10; ++k) printf("g_ni %d %.17e\n", k, g_ni[k]);
    for (int k = 0; k < 10; ++k) printf("g_nj %d %.17e\n", k, g_nj[k]);
    for (int k = 0; k < 3; ++k) printf("gdr %d %.17e\n", k, gdr[k]);
    printf("g_bi 0 %.17e\ng_bj 0 %.17e\n", g_bi, g_bj);
    double f_i[10], f_j[10];
    slater_elec_pair_field<double>(m_i, m_j, n_i, n_j, dr[0], dr[1], dr[2], b_i, b_j, f_i, f_j);
    for (int k = 0; k < 10; ++k) printf("f_i %d %.17e\n", k, f_i[k]);
    for (int k = 0; k < 10; ++k) printf("f_j %d %.17e\n", k, f_j[k]);

    // field VJP through nested duals: seed m with lambda, read the tangents
    {
        using D = Dual<double>;
        double l_i[10] = {0.7, -0.2, 0.4, 0.1, 0.05, -0.03, 0.02, 0.06, -0.01, 0.03};
        double l_j[10] = {-0.3, 0.5, 0.1, -0.6, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02};
        D mi[10], mj[10], ni[10], nj[10];
        for (int k = 0; k < 10; ++k) { mi[k] = D(m_i[k], l_i[k]); mj[k] = D(m_j[k], l_j[k]); ni[k] = D(n_i[k]); nj[k] = D(n_j[k]); }
        D e2, gmi[10], gmj[10], gni[10], gnj[10], gdr2[3], gbi2, gbj2;
        slater_elec_pair_full<D>(mi, mj, ni, nj, D(dr[0]), D(dr[1]), D(dr[2]), D(b_i), D(b_j),
                                 e2, gmi, gmj, gni, gnj, gdr2, gbi2, gbj2);
        printf("vjp_e 0 %.17e\n", e2.d);
        for (int k = 0; k < 10; ++k) printf("vjp_mi %d %.17e\n", k, gmi[k].d);
        for (int k = 0; k < 10; ++k) printf("vjp_mj %d %.17e\n", k, gmj[k].d);
        for (int k = 0; k < 10; ++k) printf("vjp_ni %d %.17e\n", k, gni[k].d);
        for (int k = 0; k < 10; ++k) printf("vjp_nj %d %.17e\n", k, gnj[k].d);
        for (int k = 0; k < 3; ++k) printf("vjp_dr %d %.17e\n", k, gdr2[k].d);
        printf("vjp_bi 0 %.17e\nvjp_bj 0 %.17e\n", gbi2.d, gbj2.d);
    }
    // energy HVP: every input seeded (the double backward of the energy op). Prints
    // tangent(e) = v . dE/d*  and  tangent(dE/d*) = H v.
    {
        using D = Dual<double>;
        double v_mi[10] = {0.7, -0.2, 0.4, 0.1, 0.05, -0.03, 0.02, 0.06, -0.01, 0.03};
        double v_mj[10] = {-0.3, 0.5, 0.1, -0.6, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02};
        double v_ni[10] = {0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double v_nj[10] = {-0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        double v_dr[3] = {0.13, -0.27, 0.05};
        double v_bi = 0.31, v_bj = -0.17;
        D mi[10], mj[10], ni[10], nj[10];
        for (int k = 0; k < 10; ++k) { mi[k] = D(m_i[k], v_mi[k]); mj[k] = D(m_j[k], v_mj[k]); ni[k] = D(n_i[k], v_ni[k]); nj[k] = D(n_j[k], v_nj[k]); }
        D e3, gmi[10], gmj[10], gni[10], gnj[10], gdr3[3], gbi3, gbj3;
        slater_elec_pair_full<D>(mi, mj, ni, nj, D(dr[0], v_dr[0]), D(dr[1], v_dr[1]), D(dr[2], v_dr[2]),
                                 D(b_i, v_bi), D(b_j, v_bj), e3, gmi, gmj, gni, gnj, gdr3, gbi3, gbj3);
        printf("hvp_e 0 %.17e\n", e3.d);
        for (int k = 0; k < 10; ++k) printf("hvp_mi %d %.17e\n", k, gmi[k].d);
        for (int k = 0; k < 10; ++k) printf("hvp_mj %d %.17e\n", k, gmj[k].d);
        for (int k = 0; k < 10; ++k) printf("hvp_ni %d %.17e\n", k, gni[k].d);
        for (int k = 0; k < 10; ++k) printf("hvp_nj %d %.17e\n", k, gnj[k].d);
        for (int k = 0; k < 3; ++k) printf("hvp_dr %d %.17e\n", k, gdr3[k].d);
        printf("hvp_bi 0 %.17e\nhvp_bj 0 %.17e\n", gbi3.d, gbj3.d);
    }
    return 0;
}
