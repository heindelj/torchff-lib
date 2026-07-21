#ifndef TORCHFF_VDW_TAPER_CUH
#define TORCHFF_VDW_TAPER_CUH

// OpenMM-compatible quintic multiplicative VdW taper (NonbondedForce / AmoebaVdwForce).
// S(r) = 1 + x^3 (C3 + x(C4 + x C5)), x = r - r_on, width = r_on - r_off (negative).
// dE_out/dr = S * dE/dr + E * dS/dr  (E is untapered energy before this call).

template <typename scalar_t>
__device__ __forceinline__ scalar_t vdw_taper_value(
    scalar_t r,
    scalar_t r_on,
    scalar_t c3,
    scalar_t c4,
    scalar_t c5
) {
    if (r <= r_on) {
        return scalar_t(1.0);
    }
    scalar_t x = r - r_on;
    return scalar_t(1.0) + x * x * x * (c3 + x * (c4 + x * c5));
}

template <typename scalar_t>
__device__ __forceinline__ void apply_vdw_taper(
    scalar_t r,
    scalar_t r_on,
    scalar_t c3,
    scalar_t c4,
    scalar_t c5,
    scalar_t& energy,
    scalar_t& dedr
) {
    if (r <= r_on) {
        return;
    }
    scalar_t x = r - r_on;
    scalar_t taper = scalar_t(1.0) + x * x * x * (c3 + x * (c4 + x * c5));
    scalar_t dtaper = x * x * (scalar_t(3.0) * c3 + x * (scalar_t(4.0) * c4 + x * scalar_t(5.0) * c5));
    scalar_t energy_untapered = energy;
    dedr = dedr * taper + energy_untapered * dtaper;
    energy *= taper;
}

#endif
