"""Slater-penetrated multipole electrostatics with per-pair output (rsfff.ff.electrostatics).

    E_ij = m_j^T T0 m_i  +  s_j^T Tss s_i  +  n_j^T T1c(b_i) s_i  +  s_j^T T1c(b_j) n_i,   s = m - n

``m`` are the full multipoles, ``n`` the nuclear point charges (a polytensor with only slot 0),
``T0`` the bare interaction tensor and the damped tensors carry pyCMM's overlap-complement
minus. Multipoles are ``(N, K)`` polytensors, ``K`` = 1, 4 or 10 -- ``[q, mu, Q]`` with the
``[1/3, 2/3, 2/3, 1/3, 2/3, 1/3]`` weights on the Cartesian quadrupole -- and every op is
per pair, gated by a caller-supplied ``(P,)`` weight.

Two ops, each with a pure-torch reference and a CUDA kernel:

* :func:`slater_elec_pair_energy` -- ``(P,)`` energies, with **double backward** through the
  kernel (the ``ffterms`` Energy/Grad pattern: the backward of the gradient op is the
  Hessian-vector product, the gradient kernel instantiated on dual numbers), so a force loss
  can be trained through it.
* :func:`slater_pauli_pair_energy` -- the Slater multipolar Pauli repulsion
  ``a_j^T f_2c(b_ij r) T a_i`` on per-pair Pauli polytensors ``(P, K)`` with one combined
  exponent per pair; the same Energy/Grad/HVP pattern.
* :func:`slater_elec_field` -- ``dE/dm`` as ``(N, K)``: the matvec of a coupled polarization
  solve. Its backward is the exact VJP with respect to every input (a nested dual-number
  evaluation of the per-pair math), which is what an adjoint solve needs.

The damped tensor here is the per-power form: ``1/r^n`` scaled by its own damping factor.
rsfff's original ``damped_interaction_tensor`` built ``r^-7`` and ``r^-9`` from the already
damped ``r^-5`` (with the wrong sign, since the callers pass ``-f``); that was fixed in rsfff
when this module was written, and the two now agree to round-off (``tests/test_slaterelec.py``).

Units are the caller's (rsfff: bohr, e, Hartree). Pair indices are ``(P, 2)``.
"""

from __future__ import annotations

import torch

try:
    import torchff_slaterelec  # noqa: F401  registers torch.ops.torchff.slater_elec_*

    HAVE_KERNELS = True
except ImportError:
    HAVE_KERNELS = False


def _register_fakes():
    """Shape-only ("fake") implementations, so ``torch.compile`` can trace through the ops.

    Dynamo needs to know every op's output shapes without running it; without these it
    graph-breaks at each kernel call and the surrounding code cannot be fused.
    """
    lib = torch.library

    @lib.register_fake("torchff::slater_elec_pair_energy")
    def _(coords, pairs, b, gate, m, n):
        return coords.new_empty(pairs.shape[0])

    @lib.register_fake("torchff::slater_elec_pair_grad")
    def _(coords, pairs, b, gate, m, n, g):
        return (torch.empty_like(coords), torch.empty_like(b), torch.empty_like(gate),
                m.new_empty(m.shape[0], 10), n.new_empty(n.shape[0], 10))

    @lib.register_fake("torchff::slater_elec_pair_hvp")
    def _(coords, pairs, b, gate, m, n, g, v_coords, v_b, v_gate, v_m, v_n):
        return (torch.empty_like(coords), torch.empty_like(b), torch.empty_like(gate),
                m.new_empty(m.shape[0], 10), n.new_empty(n.shape[0], 10), torch.empty_like(g))

    @lib.register_fake("torchff::slater_elec_field")
    def _(coords, pairs, b, gate, m, n):
        return m.new_empty(m.shape[0], 10)

    @lib.register_fake("torchff::slater_elec_field_vjp")
    def _(coords, pairs, b, gate, m, n, lam):
        return (torch.empty_like(coords), torch.empty_like(b), torch.empty_like(gate),
                m.new_empty(m.shape[0], 10), n.new_empty(n.shape[0], 10))

    @lib.register_fake("torchff::slater_pauli_pair_energy")
    def _(coords, pairs, b, a_i, a_j):
        return coords.new_empty(pairs.shape[0])

    @lib.register_fake("torchff::slater_pauli_pair_grad")
    def _(coords, pairs, b, a_i, a_j, g):
        P = pairs.shape[0]
        return (torch.empty_like(coords), torch.empty_like(b), a_i.new_empty(P, 10), a_j.new_empty(P, 10))

    @lib.register_fake("torchff::slater_pauli_pair_hvp")
    def _(coords, pairs, b, a_i, a_j, g, v_coords, v_b, v_ai, v_aj):
        P = pairs.shape[0]
        return (torch.empty_like(coords), torch.empty_like(b), a_i.new_empty(P, 10), a_j.new_empty(P, 10),
                torch.empty_like(g))


if HAVE_KERNELS:
    try:
        _register_fakes()
    except Exception:  # an older torch without register_fake: compile just graph-breaks here
        pass

__all__ = [
    "HAVE_KERNELS",
    "slater_two_center_damp",
    "slater_one_center_damp",
    "damped_interaction_tensor",
    "multipole_pair_energy",
    "slater_elec_pair_energy",
    "slater_elec_pair_energy_ref",
    "slater_elec_field",
    "slater_elec_field_ref",
    "slater_pauli_pair_energy",
    "slater_pauli_pair_energy_ref",
]


def _check_rank(max_rank: int) -> int:
    max_rank = int(max_rank)
    if max_rank in (0, 1, 2):
        return max_rank
    raise ValueError(f"max_rank must be 0, 1 or 2, got {max_rank}")


def _rank_of(m: torch.Tensor) -> int:
    return {1: 0, 4: 1, 10: 2}[int(m.shape[-1])]


# ---------------------------------------------------------------------------------------
# reference math, transcribed from rsfff.ff.multipole (per-power damping)
# ---------------------------------------------------------------------------------------

def slater_two_center_damp(u: torch.Tensor, max_rank: int = 1) -> torch.Tensor:
    """Two-center Slater damping factors, stacked ``(2*max_rank + 1, ...)``.

    Returns ``f1`` for ``max_rank=0`` and ``(f1, f3, f5)`` for ``max_rank=1``, where the
    index into the leading axis is the multipole order that factor damps: ``f1`` scales
    ``1/r``, ``f3`` scales ``1/r^3``, ``f5`` scales ``1/r^5``.

    ``u = b_ij * r`` with ``b_ij`` the combined exponent (``sqrt(b_i b_j)`` in pyCMM's
    combination rule). This is the *equal-exponent* two-center form; the polynomials
    already assume ``b_i == b_j == b_ij``, which is what makes a single combined exponent
    the right argument.
    """
    max_rank = _check_rank(max_rank)
    u2 = u * u
    u3 = u2 * u
    exp_u = torch.exp(-u)
    p1 = 1.0 + 11.0 * u / 16.0 + 3.0 * u2 / 16.0 + u3 / 48.0
    if max_rank == 0:
        return torch.stack([p1 * exp_u], dim=0)
    u4 = u3 * u
    u5 = u4 * u
    p3 = 1.0 + u + u2 / 2.0 + 7.0 * u3 / 48.0 + u4 / 48.0
    # The u^0..u^4 head shared by p5, p7 and p9; they differ only from u^5 on.
    head = 1.0 + u + u2 / 2.0 + u3 / 6.0 + u4 / 24.0
    p5 = head + u5 / 144.0
    if max_rank == 1:
        return torch.stack([p1 * exp_u, p3 * exp_u, p5 * exp_u], dim=0)
    u6 = u5 * u
    p7 = head + u5 / 120.0 + u6 / 720.0
    p9 = p7 + u6 * u / 5040.0
    return torch.stack(
        [p1 * exp_u, p3 * exp_u, p5 * exp_u, p7 * exp_u, p9 * exp_u], dim=0
    )


def slater_one_center_damp(u: torch.Tensor, max_rank: int = 1) -> torch.Tensor:
    """One-center Slater damping factors, stacked like :func:`slater_two_center_damp`.

    The point-charge/Slater-density case: one site is a point (a nucleus, or a probe), the
    other a Slater 1s density, so ``u = b_i * r`` uses that one site's exponent and no
    combination rule applies. Used by the electrostatics nucleus-shell terms rather than by
    Pauli, but it belongs with its two-center sibling.
    """
    max_rank = _check_rank(max_rank)
    u2 = u * u
    exp_u = torch.exp(-u)
    p1 = 1.0 + u / 2.0
    if max_rank == 0:
        return torch.stack([p1 * exp_u], dim=0)
    p3 = 1.0 + u + u2 / 2.0
    p5 = p3 + u2 * u / 6.0
    if max_rank == 1:
        return torch.stack([p1 * exp_u, p3 * exp_u, p5 * exp_u], dim=0)
    u4 = u2 * u2
    # Note p9 builds on p5, not on p7 (pyCMM/cmm/short_range.py:17-18) -- transcribed
    # rather than "corrected", since it is the fitted model's actual functional form.
    p7 = p5 + u4 / 30.0
    p9 = p5 + u4 * 4.0 / 105.0 + u4 * u / 210.0
    return torch.stack(
        [p1 * exp_u, p3 * exp_u, p5 * exp_u, p7 * exp_u, p9 * exp_u], dim=0
    )




def damped_interaction_tensor(
    dr_vec: torch.Tensor,                  # (P, 3) bohr, = pos[j] - pos[i]
    damp: torch.Tensor | None,             # (2*max_rank+1, P) or None for the bare tensor
    r_inv: torch.Tensor | None = None,     # (P,) 1/|dr_vec|, reused if already computed
    max_rank: int = 1,
) -> torch.Tensor:
    """Multipole interaction tensor ``(P, K, K)``, ``K = 1``, ``4`` or ``10`` by rank.

    Each inverse power is scaled by its damping factor before the Cartesian derivatives are
    assembled -- ``1/r`` by ``damp[0]``, ``1/r^3`` by ``damp[1]``, and so on up to
    ``1/r^9`` by ``damp[4]`` -- which is what makes the result the *damped* tensor rather
    than the bare one scaled uniformly. Passing ``damp=None`` gives the undamped tensor
    (useful for tests and for the long-range electrostatics).

    The rank-2 block is the third and fourth derivatives of ``1/r``; the undamped tensor is
    checked against autograd derivatives of ``1/r`` in ``tests/test_ff_multipole.py``,
    which is what makes a 100-entry transcription safe.
    """
    max_rank = _check_rank(max_rank)
    if r_inv is None:
        r_inv = 1.0 / dr_vec.norm(dim=-1)
    if damp is not None:
        expected = 2 * max_rank + 1
        if damp.shape[0] != expected:
            raise ValueError(
                f"max_rank={max_rank} needs {expected} damping factors, got {damp.shape[0]}"
            )

    r_inv1 = r_inv if damp is None else r_inv * damp[0]
    if max_rank == 0:
        return r_inv1.reshape(-1, 1, 1)

    # Every inverse power is formed from the *undamped* r_inv before any damping is applied.
    # Until the torchff port (M5) this built r_inv7 and r_inv9 from the already-damped
    # r_inv5, so the rank-2 blocks carried damp[2] * damp[3] and damp[2] * damp[4] instead
    # of damp[3] and damp[4] -- and, with the overlap-complement minus the callers pass, the
    # wrong *sign*: (-f5)(-f7) = +f5 f7 where -f7 was meant. Charge-charge, charge-dipole,
    # dipole-dipole and charge-quadrupole blocks were unaffected. Found by checking the
    # CUDA kernel (torchff csrc/slaterelec) against this function; the kernel implements the
    # documented per-power form, which is now what this function does too.
    r_inv2 = r_inv * r_inv
    r_inv3 = r_inv2 * r_inv
    r_inv5 = r_inv3 * r_inv2
    r_inv7 = r_inv5 * r_inv2
    r_inv9 = r_inv7 * r_inv2
    if damp is not None:
        r_inv3 = r_inv3 * damp[1]
        r_inv5 = r_inv5 * damp[2]

    x, y, z = dr_vec[:, 0], dr_vec[:, 1], dr_vec[:, 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    # First derivatives of 1/r: t_a = d/da (1/r) = -a/r^3.
    tx, ty, tz = -x * r_inv3, -y * r_inv3, -z * r_inv3
    # Second derivatives: t_ab = 3 a b / r^5 - delta_ab / r^3.
    txx = 3.0 * x2 * r_inv5 - r_inv3
    txy = 3.0 * xy * r_inv5
    txz = 3.0 * xz * r_inv5
    tyy = 3.0 * y2 * r_inv5 - r_inv3
    tyz = 3.0 * yz * r_inv5
    tzz = 3.0 * z2 * r_inv5 - r_inv3

    if max_rank == 1:
        # Row/column order [q, mu_x, mu_y, mu_z]; energy is m_j^T T m_i, so the
        # charge-dipole blocks carry opposite signs (pyCMM/cmm/multipole.py:321-327).
        return torch.stack(
            (
                r_inv1, -tx, -ty, -tz,
                tx, -txx, -txy, -txz,
                ty, -txy, -tyy, -tyz,
                tz, -txz, -tyz, -tzz,
            ),
            dim=-1,
        ).reshape(-1, 4, 4)

    if damp is not None:
        r_inv7 = r_inv7 * damp[3]
        r_inv9 = r_inv9 * damp[4]

    # Third derivatives: t_abc = -15 abc/r^7 + 3(a d_bc + b d_ac + c d_ab)/r^5.
    txxx = -15.0 * x2 * x * r_inv7 + 9.0 * x * r_inv5
    txxy = -15.0 * x2 * y * r_inv7 + 3.0 * y * r_inv5
    txxz = -15.0 * x2 * z * r_inv7 + 3.0 * z * r_inv5
    tyyy = -15.0 * y2 * y * r_inv7 + 9.0 * y * r_inv5
    tyyx = -15.0 * y2 * x * r_inv7 + 3.0 * x * r_inv5
    tyyz = -15.0 * y2 * z * r_inv7 + 3.0 * z * r_inv5
    tzzz = -15.0 * z2 * z * r_inv7 + 9.0 * z * r_inv5
    tzzx = -15.0 * z2 * x * r_inv7 + 3.0 * x * r_inv5
    tzzy = -15.0 * z2 * y * r_inv7 + 3.0 * y * r_inv5
    txyz = -15.0 * x * yz * r_inv7

    # Fourth derivatives.
    txxxx = 105.0 * x2 * x2 * r_inv9 - 90.0 * x2 * r_inv7 + 9.0 * r_inv5
    txxxy = 105.0 * x2 * xy * r_inv9 - 45.0 * xy * r_inv7
    txxxz = 105.0 * x2 * xz * r_inv9 - 45.0 * xz * r_inv7
    txxyy = 105.0 * x2 * y2 * r_inv9 - 15.0 * (x2 + y2) * r_inv7 + 3.0 * r_inv5
    txxzz = 105.0 * x2 * z2 * r_inv9 - 15.0 * (x2 + z2) * r_inv7 + 3.0 * r_inv5
    txxyz = 105.0 * x2 * yz * r_inv9 - 15.0 * yz * r_inv7
    tyyyy = 105.0 * y2 * y2 * r_inv9 - 90.0 * y2 * r_inv7 + 9.0 * r_inv5
    tyyyx = 105.0 * y2 * xy * r_inv9 - 45.0 * xy * r_inv7
    tyyyz = 105.0 * y2 * yz * r_inv9 - 45.0 * yz * r_inv7
    tyyzz = 105.0 * y2 * z2 * r_inv9 - 15.0 * (y2 + z2) * r_inv7 + 3.0 * r_inv5
    tyyxz = 105.0 * y2 * xz * r_inv9 - 15.0 * xz * r_inv7
    tzzzz = 105.0 * z2 * z2 * r_inv9 - 90.0 * z2 * r_inv7 + 9.0 * r_inv5
    tzzzx = 105.0 * z2 * xz * r_inv9 - 45.0 * xz * r_inv7
    tzzzy = 105.0 * z2 * yz * r_inv9 - 45.0 * yz * r_inv7
    tzzxy = 105.0 * z2 * xy * r_inv9 - 15.0 * xy * r_inv7

    # Row/column order [q, mu_x, mu_y, mu_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz],
    # transcribed from pyCMM/cmm/multipole.py:329-340.
    return torch.stack(
        (
            r_inv1, -tx, -ty, -tz, txx, txy, txz, tyy, tyz, tzz,
            tx, -txx, -txy, -txz, txxx, txxy, txxz, tyyx, txyz, tzzx,
            ty, -txy, -tyy, -tyz, txxy, tyyx, txyz, tyyy, tyyz, tzzy,
            tz, -txz, -tyz, -tzz, txxz, txyz, tzzx, tyyz, tzzy, tzzz,
            txx, -txxx, -txxy, -txxz, txxxx, txxxy, txxxz, txxyy, txxyz, txxzz,
            txy, -txxy, -tyyx, -txyz, txxxy, txxyy, txxyz, tyyyx, tyyxz, tzzxy,
            txz, -txxz, -txyz, -tzzx, txxxz, txxyz, txxzz, tyyxz, tzzxy, tzzzx,
            tyy, -tyyx, -tyyy, -tyyz, txxyy, tyyyx, tyyxz, tyyyy, tyyyz, tyyzz,
            tyz, -txyz, -tyyz, -tzzy, txxyz, tyyxz, tzzxy, tyyyz, tyyzz, tzzzy,
            tzz, -tzzx, -tzzy, -tzzz, txxzz, tzzxy, tzzzx, tyyzz, tzzzy, tzzzz,
        ),
        dim=-1,
    ).reshape(-1, 10, 10)



def multipole_pair_energy(m_i, m_j, tensor) -> torch.Tensor:
    """``m_j^T T m_i`` per pair: ``(P,)``. pyCMM's contraction order."""
    return torch.einsum("pa,pab,pb->p", m_j, tensor, m_i)


def _use_kernel(coords, use_customized_ops):
    if use_customized_ops is None:
        return HAVE_KERNELS and coords.is_cuda
    if use_customized_ops and not HAVE_KERNELS:
        raise RuntimeError("torchff_slaterelec is not compiled (CPU-only install?)")
    return bool(use_customized_ops)


def _tensors(coords, pairs, b, max_rank):
    i, j = pairs[:, 0], pairs[:, 1]
    dr = coords[j] - coords[i]
    r = dr.norm(dim=-1)
    r_inv = 1.0 / r
    b_ij = (0.5 * (b[i].log() + b[j].log())).exp()
    return (
        damped_interaction_tensor(dr, None, r_inv, max_rank=max_rank),
        damped_interaction_tensor(dr, -slater_two_center_damp(b_ij * r, max_rank), r_inv, max_rank=max_rank),
        damped_interaction_tensor(dr, -slater_one_center_damp(b[i] * r, max_rank), r_inv, max_rank=max_rank),
        damped_interaction_tensor(dr, -slater_one_center_damp(b[j] * r, max_rank), r_inv, max_rank=max_rank),
    )


def slater_elec_pair_energy_ref(coords, pairs, b, gate, m, n) -> torch.Tensor:
    """``(P,)`` reference: ``gate * (point + penetration)`` per pair."""
    max_rank = _rank_of(m)
    i, j = pairs[:, 0], pairs[:, 1]
    t0, tss, t1i, t1j = _tensors(coords, pairs, b, max_rank)
    s = m - n
    e = (
        multipole_pair_energy(m[i], m[j], t0)
        + multipole_pair_energy(s[i], s[j], tss)
        + multipole_pair_energy(s[i], n[j], t1i)
        + multipole_pair_energy(n[i], s[j], t1j)
    )
    return gate * e


def slater_elec_field_ref(coords, pairs, b, gate, m, n) -> torch.Tensor:
    """``(N, K)`` reference: ``d(sum_p e_p)/dm`` written out as the two-sided contraction."""
    max_rank = _rank_of(m)
    i, j = pairs[:, 0], pairs[:, 1]
    t0, tss, t1i, t1j = _tensors(coords, pairs, b, max_rank)
    w = gate[:, None, None]
    t0, tss, t1i, t1j = w * t0, w * tss, w * t1i, w * t1j
    s = m - n
    g_i = (
        torch.einsum("pab,pa->pb", t0, m[j])
        + torch.einsum("pab,pa->pb", tss, s[j])
        + torch.einsum("pab,pa->pb", t1i, n[j])
    )
    g_j = (
        torch.einsum("pab,pb->pa", t0, m[i])
        + torch.einsum("pab,pb->pa", tss, s[i])
        + torch.einsum("pab,pb->pa", t1j, n[i])
    )
    out = torch.zeros_like(m)
    return out.index_add(0, i, g_i).index_add(0, j, g_j)


# ---------------------------------------------------------------------------------------
# kernels + autograd
# ---------------------------------------------------------------------------------------

class _EnergyGrad(torch.autograd.Function):
    """``(coords, b, gate, m, n, g) -> g-weighted first derivatives``; its backward is the HVP.

    The ``ffterms`` pattern: ``_Energy.backward`` calls this instead of scaling saved tensors,
    so a force loss can be backpropagated through the kernel (``create_graph=True``). The
    backward instantiates the per-pair math on ``Dual`` with the cotangents as tangent seeds
    (``energy_hvp_kernel``).
    """

    @staticmethod
    def forward(ctx, coords, pairs, b, gate, m, n, g):
        K = m.shape[1]
        d_coords, d_b, d_gate, d_m, d_n = torch.ops.torchff.slater_elec_pair_grad(
            coords, pairs, b, gate, m, n, g
        )
        ctx.save_for_backward(coords, pairs, b, gate, m, n, g)
        return d_coords, d_b, d_gate, d_m[:, :K].contiguous(), d_n[:, :K].contiguous()

    @staticmethod
    def backward(ctx, v_coords, v_b, v_gate, v_m, v_n):
        coords, pairs, b, gate, m, n, g = ctx.saved_tensors
        K = m.shape[1]
        v_coords = torch.zeros_like(coords) if v_coords is None else v_coords.contiguous()
        v_b = torch.zeros_like(b) if v_b is None else v_b.contiguous()
        v_gate = torch.zeros_like(gate) if v_gate is None else v_gate.contiguous()
        v_m = torch.zeros_like(m) if v_m is None else v_m.contiguous()
        v_n = torch.zeros_like(n) if v_n is None else v_n.contiguous()
        o_coords, o_b, o_gate, o_m, o_n, o_g = torch.ops.torchff.slater_elec_pair_hvp(
            coords, pairs, b, gate, m, n, g, v_coords, v_b, v_gate, v_m, v_n
        )
        return o_coords, None, o_b, o_gate, o_m[:, :K], o_n[:, :K], o_g


class _Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, b, gate, m, n):
        ctx.save_for_backward(coords, pairs, b, gate, m, n)
        return torch.ops.torchff.slater_elec_pair_energy(coords, pairs, b, gate, m, n)

    @staticmethod
    def backward(ctx, g):
        coords, pairs, b, gate, m, n = ctx.saved_tensors
        d_coords, d_b, d_gate, d_m, d_n = _EnergyGrad.apply(
            coords, pairs, b, gate, m, n, g.contiguous()
        )
        return d_coords, None, d_b, d_gate, d_m, d_n


class _Field(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, b, gate, m, n):
        ctx.save_for_backward(coords, pairs, b, gate, m, n)
        K = m.shape[1]
        return torch.ops.torchff.slater_elec_field(coords, pairs, b, gate, m, n)[:, :K]

    @staticmethod
    def backward(ctx, lam):
        coords, pairs, b, gate, m, n = ctx.saved_tensors
        d_coords, d_b, d_gate, d_m, d_n = torch.ops.torchff.slater_elec_field_vjp(
            coords, pairs, b, gate, m, n, lam.contiguous()
        )
        K = m.shape[1]
        return d_coords, None, d_b, d_gate, d_m[:, :K], d_n[:, :K]


def _prep(coords, pairs, b, gate, m, n):
    return (coords.contiguous(), pairs.contiguous(), b.contiguous(), gate.contiguous(),
            m.contiguous(), n.contiguous())


def slater_elec_pair_energy(coords, pairs, b, gate, m, n, *, use_customized_ops=None):
    """``(P,)`` gated Slater-penetrated multipole energies; double backward on CUDA too."""
    if _use_kernel(coords, use_customized_ops):
        return _Energy.apply(*_prep(coords, pairs, b, gate, m, n))
    return slater_elec_pair_energy_ref(coords, pairs, b, gate, m, n)


def slater_elec_field(coords, pairs, b, gate, m, n, *, use_customized_ops=None):
    """``(N, K)`` field ``d(sum e)/dm``: the polarization matvec, with an exact first-order VJP."""
    if _use_kernel(coords, use_customized_ops):
        return _Field.apply(*_prep(coords, pairs, b, gate, m, n))
    return slater_elec_field_ref(coords, pairs, b, gate, m, n)


# ---------------------------------------------------------------------------------------
# Pauli repulsion
# ---------------------------------------------------------------------------------------

def slater_pauli_pair_energy_ref(coords, pairs, b_ij, a_i, a_j) -> torch.Tensor:
    """``(P,)`` reference: ``a_j^T [f_2c(b_ij r) T] a_i`` on per-pair polytensors."""
    max_rank = _rank_of(a_i)
    dr = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    r = dr.norm(dim=-1)
    damp = slater_two_center_damp(b_ij * r, max_rank)
    t = damped_interaction_tensor(dr, damp, 1.0 / r, max_rank=max_rank)
    return multipole_pair_energy(a_i, a_j, t)


class _PauliGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, b, a_i, a_j, g):
        K = a_i.shape[1]
        d_coords, d_b, d_ai, d_aj = torch.ops.torchff.slater_pauli_pair_grad(coords, pairs, b, a_i, a_j, g)
        ctx.save_for_backward(coords, pairs, b, a_i, a_j, g)
        return d_coords, d_b, d_ai[:, :K].contiguous(), d_aj[:, :K].contiguous()

    @staticmethod
    def backward(ctx, v_coords, v_b, v_ai, v_aj):
        coords, pairs, b, a_i, a_j, g = ctx.saved_tensors
        K = a_i.shape[1]
        v_coords = torch.zeros_like(coords) if v_coords is None else v_coords.contiguous()
        v_b = torch.zeros_like(b) if v_b is None else v_b.contiguous()
        v_ai = torch.zeros_like(a_i) if v_ai is None else v_ai.contiguous()
        v_aj = torch.zeros_like(a_j) if v_aj is None else v_aj.contiguous()
        o_coords, o_b, o_ai, o_aj, o_g = torch.ops.torchff.slater_pauli_pair_hvp(
            coords, pairs, b, a_i, a_j, g, v_coords, v_b, v_ai, v_aj
        )
        return o_coords, None, o_b, o_ai[:, :K], o_aj[:, :K], o_g


class _Pauli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, b, a_i, a_j):
        ctx.save_for_backward(coords, pairs, b, a_i, a_j)
        return torch.ops.torchff.slater_pauli_pair_energy(coords, pairs, b, a_i, a_j)

    @staticmethod
    def backward(ctx, g):
        coords, pairs, b, a_i, a_j = ctx.saved_tensors
        d_coords, d_b, d_ai, d_aj = _PauliGrad.apply(coords, pairs, b, a_i, a_j, g.contiguous())
        return d_coords, None, d_b, d_ai, d_aj


def slater_pauli_pair_energy(coords, pairs, b_ij, a_i, a_j, *, use_customized_ops=None):
    """``(P,)`` Slater multipolar Pauli energies; ``a_i``/``a_j`` are ``(P, K)`` per-pair Pauli
    polytensors and ``b_ij`` the ``(P,)`` combined exponent. Double backward on CUDA."""
    if _use_kernel(coords, use_customized_ops):
        return _Pauli.apply(coords.contiguous(), pairs.contiguous(), b_ij.contiguous(),
                            a_i.contiguous(), a_j.contiguous())
    return slater_pauli_pair_energy_ref(coords, pairs, b_ij, a_i, a_j)
