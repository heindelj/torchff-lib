"""Per-term force-field kernels with the *per-pair output, double-backward* convention.

These are the terms a neural network emits parameters for (the rsfff "force field
functional"): every parameter is a tensor with a gradient, the energy of each pair / bond /
angle is returned **unreduced** so the caller can gate, weight and pool it however its
bookkeeping needs, and every op supports ``create_graph=True`` so a force loss
``|| -dE/dR - F_ref ||`` can be backpropagated into the parameters.

Conventions shared by every op in this module
---------------------------------------------
* ``coords`` is ``(N, 3)``; index tensors are ``(P, 2)`` / ``(Nb, 2)`` / ``(Na, 3)`` int64,
  matching the rest of torchff (``pairs[:, 0] = i``, ``pairs[:, 1] = j``).
* Parameters are **per element of the index tensor** (per pair, per bond, per angle),
  already combined by whatever rule the caller wants (geometric mean, table lookup, ...).
  Combination rules stay in torch: they are cheap, and keeping them out of the kernel is
  what lets one kernel serve every parameterisation.
* Units are the caller's. Nothing here converts; ``coords`` and the parameters must agree.
* Output is ``(P,)`` (resp. ``(Nb,)``, ``(Na,)``), never a scalar. No cutoff is applied
  inside the op -- the pair list already is the cutoff, and a smooth switch belongs to the
  caller.
* Each op comes as a pure-torch ``*_ref`` function (the formula, runnable anywhere) and a
  dispatcher that uses the CUDA kernels when they are compiled and the inputs are on a CUDA
  device. The two agree to round-off, including second derivatives
  (``tests/test_ffterms.py``).

How double backward works here
------------------------------
Each CUDA op is a pair of ``autograd.Function``\\ s. ``_Energy.forward`` runs the energy
kernel; its ``backward`` does *not* scale saved gradients (that would be a dead end for
autograd) but calls ``_Grad.apply``, whose forward is the gradient kernel
``(coords, params, g) -> (dE/dcoords, dE/dparams) * g`` and whose backward is the
Hessian-vector-product kernel. The HVP kernels are the gradient kernels instantiated on a
dual-number scalar type (``csrc/common/dual.cuh``), so there is one hand-written derivative
per term and the second order comes for free.
"""

from __future__ import annotations

import math

import torch

try:  # registers torch.ops.torchff.*_pair_energy / _grad / _hvp
    import torchff_ffterms  # noqa: F401

    HAVE_KERNELS = True
except ImportError:  # CPU-only install (TORCHFF_NO_CUDA=1)
    HAVE_KERNELS = False


def _register_fakes():
    """Shape-only implementations so ``torch.compile`` can trace through the ops."""
    lib = torch.library

    @lib.register_fake("torchff::tt_dispersion_pair_energy")
    def _(coords, pairs, c6, b):
        return coords.new_empty(pairs.shape[0])

    @lib.register_fake("torchff::tt_dispersion_pair_grad")
    def _(coords, pairs, c6, b, g):
        return torch.empty_like(coords), torch.empty_like(c6), torch.empty_like(b)

    @lib.register_fake("torchff::tt_dispersion_pair_hvp")
    def _(coords, pairs, c6, b, g, v_coords, v_c6, v_b):
        return torch.empty_like(coords), torch.empty_like(c6), torch.empty_like(b), torch.empty_like(g)

    @lib.register_fake("torchff::morse_bond_energy")
    def _(coords, bonds, r_eq, d, k):
        return coords.new_empty(bonds.shape[0])

    @lib.register_fake("torchff::morse_bond_grad")
    def _(coords, bonds, r_eq, d, k, g):
        return torch.empty_like(coords), torch.empty_like(r_eq), torch.empty_like(d), torch.empty_like(k)

    @lib.register_fake("torchff::morse_bond_hvp")
    def _(coords, bonds, r_eq, d, k, g, v_coords, v_req, v_d, v_k):
        return (torch.empty_like(coords), torch.empty_like(r_eq), torch.empty_like(d), torch.empty_like(k),
                torch.empty_like(g))

    @lib.register_fake("torchff::cosine_angle_energy")
    def _(coords, angles, cos_eq, k):
        return coords.new_empty(angles.shape[0])

    @lib.register_fake("torchff::cosine_angle_grad")
    def _(coords, angles, cos_eq, k, g):
        return torch.empty_like(coords), torch.empty_like(cos_eq), torch.empty_like(k)

    @lib.register_fake("torchff::cosine_angle_hvp")
    def _(coords, angles, cos_eq, k, g, v_coords, v_cos, v_k):
        return torch.empty_like(coords), torch.empty_like(cos_eq), torch.empty_like(k), torch.empty_like(g)


if HAVE_KERNELS:
    try:
        _register_fakes()
    except Exception:
        pass

__all__ = [
    "HAVE_KERNELS",
    "tang_toennies",
    "tt_dispersion_pair_energy",
    "tt_dispersion_pair_energy_ref",
    "morse_bond_energy",
    "morse_bond_energy_ref",
    "cosine_angle_energy",
    "cosine_angle_energy_ref",
]

#: Below this ``x = b r`` the direct Tang-Toennies form cancels catastrophically and the
#: tail series is used instead (same constant as the CUDA kernel, ``tang_toennies.cuh``).
TT_SERIES_BELOW = 2.0
_TT_SERIES_TERMS = 20


def _use_kernel(coords: torch.Tensor, use_customized_ops: bool | None) -> bool:
    if use_customized_ops is None:
        return HAVE_KERNELS and coords.is_cuda
    if use_customized_ops and not HAVE_KERNELS:
        raise RuntimeError("torchff_ffterms is not compiled (CPU-only install?)")
    return bool(use_customized_ops)


# ======================================================================================
# Tang-Toennies damped C6 dispersion:  E_ij = -f_6(b_ij r) C6_ij / r^6
# ======================================================================================

def _tt_direct(x: torch.Tensor, order: int) -> torch.Tensor:
    poly = torch.full_like(x, 1.0 / math.factorial(order))
    for k in range(order - 1, -1, -1):
        poly = poly * x + 1.0 / math.factorial(k)
    return 1.0 - torch.exp(-x) * poly


def _tt_series(x: torch.Tensor, order: int) -> torch.Tensor:
    lo = order + 1
    poly = torch.full_like(x, 1.0 / math.factorial(lo + _TT_SERIES_TERMS - 1))
    for k in range(lo + _TT_SERIES_TERMS - 2, lo - 1, -1):
        poly = poly * x + 1.0 / math.factorial(k)
    return torch.exp(-x) * poly * x.pow(lo)


def tang_toennies(x: torch.Tensor, order: int = 6, *, series_below: float = TT_SERIES_BELOW):
    """``f_n(x) = 1 - e^{-x} sum_{k<=n} x^k/k!``, cancellation-free below ``series_below``.

    The direct form subtracts two O(1) numbers to get ``~x^(n+1)/(n+1)!``; at ``x = 0.1``
    in float64 that is already 1e-5 relative error and it poisons the gradient the same
    way. Below the threshold the algebraically identical tail series
    ``e^{-x} sum_{k>n} x^k/k!`` is used instead. Both branches of ``torch.where`` are
    evaluated, so each gets a clamped argument it can handle.
    """
    xs = x.clamp(max=series_below)
    xl = x.clamp(min=series_below)
    return torch.where(x < series_below, _tt_series(xs, order), _tt_direct(xl, order))


def tt_dispersion_pair_energy_ref(
    coords: torch.Tensor, pairs: torch.Tensor, c6: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """Reference ``(P,)`` energies: ``-f_6(b_ij r_ij) c6_ij / r_ij^6``."""
    r = (coords[pairs[:, 1]] - coords[pairs[:, 0]]).norm(dim=-1)
    return -tang_toennies(b * r, 6) * c6 / r.pow(6)


class _TTDispersionGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, c6, b, g):
        d_coords, d_c6, d_b = torch.ops.torchff.tt_dispersion_pair_grad(coords, pairs, c6, b, g)
        ctx.save_for_backward(coords, pairs, c6, b, g)
        return d_coords, d_c6, d_b

    @staticmethod
    def backward(ctx, v_coords, v_c6, v_b):
        coords, pairs, c6, b, g = ctx.saved_tensors
        v_coords = torch.zeros_like(coords) if v_coords is None else v_coords
        v_c6 = torch.zeros_like(c6) if v_c6 is None else v_c6
        v_b = torch.zeros_like(b) if v_b is None else v_b
        g_coords, g_c6, g_b, g_g = torch.ops.torchff.tt_dispersion_pair_hvp(
            coords, pairs, c6, b, g, v_coords.contiguous(), v_c6.contiguous(), v_b.contiguous()
        )
        return g_coords, None, g_c6, g_b, g_g


class _TTDispersionEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, pairs, c6, b):
        ctx.save_for_backward(coords, pairs, c6, b)
        return torch.ops.torchff.tt_dispersion_pair_energy(coords, pairs, c6, b)

    @staticmethod
    def backward(ctx, g):
        coords, pairs, c6, b = ctx.saved_tensors
        d_coords, d_c6, d_b = _TTDispersionGrad.apply(coords, pairs, c6, b, g.contiguous())
        return d_coords, None, d_c6, d_b


def tt_dispersion_pair_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    c6: torch.Tensor,
    b: torch.Tensor,
    *,
    use_customized_ops: bool | None = None,
) -> torch.Tensor:
    """``(P,)`` Tang-Toennies damped C6 energies; CUDA kernel when available, else the ref.

    ``c6`` and ``b`` are per pair (``(P,)``), in units consistent with ``coords``.
    """
    if _use_kernel(coords, use_customized_ops):
        return _TTDispersionEnergy.apply(
            coords.contiguous(), pairs.contiguous(), c6.contiguous(), b.contiguous()
        )
    return tt_dispersion_pair_energy_ref(coords, pairs, c6, b)


# ======================================================================================
# Morse bond, well-referenced:  E = D [(1 - e^{-beta (r - r_eq)})^2 - 1],  beta = sqrt(k / 2D)
# ======================================================================================

def morse_bond_energy_ref(
    coords: torch.Tensor, bonds: torch.Tensor, r_eq: torch.Tensor, d: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Reference ``(Nb,)`` Morse energies, zero at dissociation and ``-D`` at the minimum.

    ``k`` is the harmonic force constant at the minimum, so ``beta = sqrt(k / 2D)``.
    Differs from :func:`torchff.bond.compute_morse_bond_energy_ref` by the ``-D`` offset
    and by returning per-bond values.
    """
    r = (coords[bonds[:, 1]] - coords[bonds[:, 0]]).norm(dim=-1)
    beta = torch.sqrt(k / (2.0 * d))
    x = 1.0 - torch.exp(-beta * (r - r_eq))
    return d * (x * x - 1.0)


class _MorseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, bonds, r_eq, d, k, g):
        out = torch.ops.torchff.morse_bond_grad(coords, bonds, r_eq, d, k, g)
        ctx.save_for_backward(coords, bonds, r_eq, d, k, g)
        return out

    @staticmethod
    def backward(ctx, v_coords, v_req, v_d, v_k):
        coords, bonds, r_eq, d, k, g = ctx.saved_tensors
        v_coords = torch.zeros_like(coords) if v_coords is None else v_coords
        v_req = torch.zeros_like(r_eq) if v_req is None else v_req
        v_d = torch.zeros_like(d) if v_d is None else v_d
        v_k = torch.zeros_like(k) if v_k is None else v_k
        g_coords, g_req, g_d, g_k, g_g = torch.ops.torchff.morse_bond_hvp(
            coords, bonds, r_eq, d, k, g,
            v_coords.contiguous(), v_req.contiguous(), v_d.contiguous(), v_k.contiguous(),
        )
        return g_coords, None, g_req, g_d, g_k, g_g


class _MorseEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, bonds, r_eq, d, k):
        ctx.save_for_backward(coords, bonds, r_eq, d, k)
        return torch.ops.torchff.morse_bond_energy(coords, bonds, r_eq, d, k)

    @staticmethod
    def backward(ctx, g):
        coords, bonds, r_eq, d, k = ctx.saved_tensors
        d_coords, d_req, d_d, d_k = _MorseGrad.apply(coords, bonds, r_eq, d, k, g.contiguous())
        return d_coords, None, d_req, d_d, d_k


def morse_bond_energy(
    coords: torch.Tensor,
    bonds: torch.Tensor,
    r_eq: torch.Tensor,
    d: torch.Tensor,
    k: torch.Tensor,
    *,
    use_customized_ops: bool | None = None,
) -> torch.Tensor:
    """``(Nb,)`` well-referenced Morse energies; CUDA kernel when available, else the ref."""
    if _use_kernel(coords, use_customized_ops):
        return _MorseEnergy.apply(
            coords.contiguous(), bonds.contiguous(),
            r_eq.contiguous(), d.contiguous(), k.contiguous(),
        )
    return morse_bond_energy_ref(coords, bonds, r_eq, d, k)


# ======================================================================================
# Cosine-harmonic angle:  E = k/2 (cos theta - cos theta_eq)^2,   angles = [i, apex, k]
# ======================================================================================

def cosine_angle_energy_ref(
    coords: torch.Tensor, angles: torch.Tensor, cos_eq: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Reference ``(Na,)`` energies on cosines throughout: smooth through linear geometries.

    Takes ``cos theta_eq`` rather than ``theta_eq`` (unlike
    :func:`torchff.angle.compute_cosine_angle_energy_ref`), because that is the quantity a
    parameter network emits without an ``acos``.
    """
    v1 = coords[angles[:, 0]] - coords[angles[:, 1]]
    v2 = coords[angles[:, 2]] - coords[angles[:, 1]]
    cos_theta = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1))
    diff = cos_theta - cos_eq
    return 0.5 * k * diff * diff


class _CosineAngleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, angles, cos_eq, k, g):
        out = torch.ops.torchff.cosine_angle_grad(coords, angles, cos_eq, k, g)
        ctx.save_for_backward(coords, angles, cos_eq, k, g)
        return out

    @staticmethod
    def backward(ctx, v_coords, v_cos, v_k):
        coords, angles, cos_eq, k, g = ctx.saved_tensors
        v_coords = torch.zeros_like(coords) if v_coords is None else v_coords
        v_cos = torch.zeros_like(cos_eq) if v_cos is None else v_cos
        v_k = torch.zeros_like(k) if v_k is None else v_k
        g_coords, g_cos, g_k, g_g = torch.ops.torchff.cosine_angle_hvp(
            coords, angles, cos_eq, k, g, v_coords.contiguous(), v_cos.contiguous(), v_k.contiguous()
        )
        return g_coords, None, g_cos, g_k, g_g


class _CosineAngleEnergy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, angles, cos_eq, k):
        ctx.save_for_backward(coords, angles, cos_eq, k)
        return torch.ops.torchff.cosine_angle_energy(coords, angles, cos_eq, k)

    @staticmethod
    def backward(ctx, g):
        coords, angles, cos_eq, k = ctx.saved_tensors
        d_coords, d_cos, d_k = _CosineAngleGrad.apply(coords, angles, cos_eq, k, g.contiguous())
        return d_coords, None, d_cos, d_k


def cosine_angle_energy(
    coords: torch.Tensor,
    angles: torch.Tensor,
    cos_eq: torch.Tensor,
    k: torch.Tensor,
    *,
    use_customized_ops: bool | None = None,
) -> torch.Tensor:
    """``(Na,)`` cosine-harmonic angle energies; CUDA kernel when available, else the ref."""
    if _use_kernel(coords, use_customized_ops):
        return _CosineAngleEnergy.apply(
            coords.contiguous(), angles.contiguous(), cos_eq.contiguous(), k.contiguous()
        )
    return cosine_angle_energy_ref(coords, angles, cos_eq, k)
