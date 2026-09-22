"""Regenerate csrc/slaterelec/multipole_pair.cuh from csrc/cmm/multipoles.cuh.

Takes the `pairwise_multipole_kernel_with_grad` overload that returns dr gradients (but no
interaction tensor) and templates it on an arbitrary scalar type T so it can run on
Dual<scalar> (common/dual.cuh). Run from the repository root after editing the CMM source.
"""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = (root / "csrc/cmm/multipoles.cuh").read_text().splitlines()

# locate the overload: the one whose signature has drx_g but not interaction_tensor
starts = [i for i, l in enumerate(src) if l.startswith("__device__ __forceinline__ void pairwise_multipole_kernel_with_grad(")]
chosen = None
for s in starts:
    sig = "\n".join(src[s:s + 25])
    if "drx_g" in sig and "interaction_tensor" not in sig:
        chosen = s
        break
assert chosen is not None
end = next(i for i in range(chosen, len(src)) if src[i] == "}")
body = "\n".join(src[chosen - 1:end + 1])
body = body.replace("scalar_t", "T").replace("rsqrt_(", "d_rsqrt(")
body = body.replace("pairwise_multipole_kernel_with_grad", "mp_pair_grad")

header = '''#ifndef TORCHFF_SLATER_ELEC_MULTIPOLE_PAIR_CUH
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

'''
(root / "csrc/slaterelec/multipole_pair.cuh").write_text(header + body + "\n\n#endif\n")
print("wrote csrc/slaterelec/multipole_pair.cuh")
