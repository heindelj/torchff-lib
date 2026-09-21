"""torchff.ffterms: per-element terms with double backward.

Three layers of checks:

1. the pure-torch reference functions against the library's older references (same formula,
   different signature) -- runs anywhere;
2. the *device math* (csrc/ffterms/terms.cuh on Dual<double>) against torch double backward,
   via a host build of the header with g++ -- runs anywhere with a C++ compiler, no CUDA;
3. the compiled CUDA ops against the references, including gradcheck / gradgradcheck -- CUDA only.
"""

import math
import os
import shutil
import subprocess
import sys

import pytest
import torch

from torchff import ffterms

torch.manual_seed(0)
DT = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(os.path.dirname(HERE), "csrc")


def _system(n=12, device="cpu"):
    coords = (torch.rand(n, 3, dtype=DT, device=device) * 3.0).requires_grad_(True)
    ii, jj = torch.triu_indices(n, n, 1)
    pairs = torch.stack([ii, jj], dim=1).to(device)
    P = pairs.shape[0]
    c6 = (torch.rand(P, dtype=DT, device=device) * 20 + 1).requires_grad_(True)
    b = (torch.rand(P, dtype=DT, device=device) * 2 + 0.3).requires_grad_(True)
    bonds = pairs[: n - 1]  # every bond from atom 0: (0, 1) ... (0, n-1)
    r_eq = (torch.rand(n - 1, dtype=DT, device=device) + 1.5).requires_grad_(True)
    d = (torch.rand(n - 1, dtype=DT, device=device) * 0.2 + 0.1).requires_grad_(True)
    k = (torch.rand(n - 1, dtype=DT, device=device) * 0.5 + 0.3).requires_grad_(True)
    angles = torch.stack([torch.arange(n - 2), torch.arange(1, n - 1), torch.arange(2, n)], 1).to(device)
    cos_eq = (torch.rand(n - 2, dtype=DT, device=device) * 0.8 - 0.4).requires_grad_(True)
    ka = (torch.rand(n - 2, dtype=DT, device=device) * 0.3 + 0.1).requires_grad_(True)
    return dict(coords=coords, pairs=pairs, c6=c6, b=b, bonds=bonds, r_eq=r_eq, d=d, k=k,
                angles=angles, cos_eq=cos_eq, ka=ka)


# --- 1. references ---------------------------------------------------------------------------

def test_tang_toennies_branches_agree_and_limit():
    x = torch.linspace(0.05, 6.0, 200, dtype=DT)
    direct = ffterms._tt_direct(x, 6)
    series = ffterms._tt_series(x, 6)
    # away from the cancellation region both forms agree to round-off
    mid = (x > 1.0) & (x < 3.0)
    assert torch.allclose(direct[mid], series[mid], rtol=1e-12, atol=1e-14)
    small = torch.tensor([1e-3, 1e-2], dtype=DT)
    assert torch.allclose(ffterms.tang_toennies(small), small ** 7 / math.factorial(7), rtol=1e-3)


def test_refs_match_legacy_torchff_formulas():
    s = _system()
    r = (s["coords"][s["pairs"][:, 1]] - s["coords"][s["pairs"][:, 0]]).norm(dim=-1)
    try:
        from torchff.dispersion import compute_tang_tonnies_dispersion_energy_ref
    except ImportError:
        pytest.skip("torchff.dispersion needs the compiled extension")
    legacy = compute_tang_tonnies_dispersion_energy_ref(r, s["c6"], s["b"], sum=False)
    mine = ffterms.tt_dispersion_pair_energy_ref(s["coords"], s["pairs"], s["c6"], s["b"])
    assert torch.allclose(mine, legacy, rtol=1e-10, atol=1e-14)


def test_morse_ref_is_well_referenced():
    s = _system()
    e = ffterms.morse_bond_energy_ref(s["coords"], s["bonds"], s["r_eq"], s["d"], s["k"])
    far = s["coords"].detach().clone()
    far[s["bonds"][:, 1]] += 1e3
    e_far = ffterms.morse_bond_energy_ref(far, s["bonds"], s["r_eq"], s["d"], s["k"])
    assert torch.allclose(e_far, torch.zeros_like(e_far), atol=1e-12)
    at_min = ffterms.morse_bond_energy_ref(
        torch.stack([torch.zeros(3, dtype=DT), torch.tensor([1.5, 0, 0], dtype=DT)]),
        torch.tensor([[0, 1]]), torch.tensor([1.5], dtype=DT), torch.tensor([0.2], dtype=DT), torch.tensor([0.5], dtype=DT))
    assert torch.allclose(at_min, torch.tensor([-0.2], dtype=DT))


def test_refs_gradcheck():
    s = _system(6)
    assert torch.autograd.gradcheck(
        lambda c, c6, b: ffterms.tt_dispersion_pair_energy_ref(c, s["pairs"], c6, b), (s["coords"], s["c6"], s["b"]))
    assert torch.autograd.gradgradcheck(
        lambda c, c6, b: ffterms.tt_dispersion_pair_energy_ref(c, s["pairs"], c6, b), (s["coords"], s["c6"], s["b"]))
    assert torch.autograd.gradgradcheck(
        lambda c, r0, d, k: ffterms.morse_bond_energy_ref(c, s["bonds"], r0, d, k), (s["coords"], s["r_eq"], s["d"], s["k"]))
    assert torch.autograd.gradgradcheck(
        lambda c, c0, k: ffterms.cosine_angle_energy_ref(c, s["angles"], c0, k), (s["coords"], s["cos_eq"], s["ka"]))


# --- 2. the device math, built for the host --------------------------------------------------

@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_device_math_matches_torch_double_backward(tmp_path):
    exe = tmp_path / "check_terms"
    subprocess.run(
        ["g++", "-std=c++17", "-O1", "-I", os.path.join(HERE, "host"), "-I", CSRC,
         os.path.join(HERE, "host", "check_terms.cpp"), "-o", str(exe)], check=True)
    out = {}
    for line in subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines():
        tag, key, v, dv = line.split()
        out.setdefault(tag, {})[key] = (float(v), float(dv))

    def compare(tag, f, x, v):
        x = torch.tensor(x, dtype=DT, requires_grad=True)
        v = torch.tensor(v, dtype=DT)
        e = f(x)
        (g,) = torch.autograd.grad(e, x, create_graph=True)
        hv = torch.autograd.grad(g @ v, x)[0]
        ref = out[tag]
        assert abs(ref["e"][0] - e.item()) < 1e-14 * max(1.0, abs(e.item()))
        assert abs(ref["e"][1] - (g @ v).item()) < 1e-13 * max(1.0, abs((g @ v).item()))
        for kk in range(x.numel()):
            gv, hvv = ref[f"g{kk}"]
            assert abs(gv - g[kk].item()) < 1e-13 * max(1.0, abs(gv)), (tag, kk, "grad")
            assert abs(hvv - hv[kk].item()) < 1e-12 * max(1.0, abs(hvv)), (tag, kk, "hvp")

    def tt(x):
        r = x[:3].norm()
        return -ffterms.tang_toennies(x[4] * r, 6) * x[3] / r ** 6

    def mo(y):
        r = y[:3].norm()
        beta = torch.sqrt(y[5] / (2 * y[4]))
        xx = 1 - torch.exp(-beta * (r - y[3]))
        return y[4] * (xx * xx - 1)

    def an(y):
        v1, v2 = y[:3], y[3:6]
        c = (v1 @ v2) / (v1.norm() * v2.norm())
        return 0.5 * y[7] * (c - y[6]) ** 2

    compare("TT0", tt, [1.3, -0.7, 2.1, 30.0, 1.7], [0.3, 0.2, -0.5, 0.7, -0.4])
    compare("TT1", tt, [1.3, -0.7, 2.1, 30.0, 0.5], [0.3, 0.2, -0.5, 0.7, -0.4])
    compare("MO", mo, [0.9, -0.4, 1.2, 1.81, 0.2, 0.54], [0.1, 0.3, -0.2, 0.05, 0.02, 0.03])
    compare("AN", an, [1.0, 0.2, -0.3, -0.4, 1.1, 0.5, -0.25, 0.17], [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.04, 0.02])


# --- 3. the CUDA ops ----------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and ffterms.HAVE_KERNELS), reason="needs CUDA + torchff_ffterms")


@needs_cuda
@pytest.mark.parametrize("term", ["tt", "morse", "angle"])
def test_kernel_matches_ref_to_second_order(term):
    s = _system(10, device="cuda")
    if term == "tt":
        args = (s["coords"], s["c6"], s["b"])
        ref = lambda c, c6, b: ffterms.tt_dispersion_pair_energy_ref(c, s["pairs"], c6, b)
        op = lambda c, c6, b: ffterms.tt_dispersion_pair_energy(c, s["pairs"], c6, b, use_customized_ops=True)
    elif term == "morse":
        args = (s["coords"], s["r_eq"], s["d"], s["k"])
        ref = lambda c, r0, d, k: ffterms.morse_bond_energy_ref(c, s["bonds"], r0, d, k)
        op = lambda c, r0, d, k: ffterms.morse_bond_energy(c, s["bonds"], r0, d, k, use_customized_ops=True)
    else:
        args = (s["coords"], s["cos_eq"], s["ka"])
        ref = lambda c, c0, k: ffterms.cosine_angle_energy_ref(c, s["angles"], c0, k)
        op = lambda c, c0, k: ffterms.cosine_angle_energy(c, s["angles"], c0, k, use_customized_ops=True)

    e_ref, e_op = ref(*args), op(*args)
    assert torch.allclose(e_op, e_ref, rtol=1e-12, atol=1e-14)

    # first order, every input
    w = torch.randn_like(e_ref)
    g_ref = torch.autograd.grad((e_ref * w).sum(), args, create_graph=True)
    g_op = torch.autograd.grad((e_op * w).sum(), args, create_graph=True)
    for a, b_ in zip(g_ref, g_op):
        assert torch.allclose(a, b_, rtol=1e-10, atol=1e-13)

    # second order: the force-loss pattern  d/dparams || dE/dcoords ||^2
    loss_ref = sum((gr ** 2).sum() for gr in g_ref)
    loss_op = sum((go ** 2).sum() for go in g_op)
    h_ref = torch.autograd.grad(loss_ref, args)
    h_op = torch.autograd.grad(loss_op, args)
    for a, b_ in zip(h_ref, h_op):
        assert torch.allclose(a, b_, rtol=1e-9, atol=1e-12)

    assert torch.autograd.gradcheck(op, args)
    assert torch.autograd.gradgradcheck(op, args)


@needs_cuda
def test_kernel_empty_inputs():
    c = torch.rand(4, 3, dtype=DT, device="cuda", requires_grad=True)
    e = ffterms.tt_dispersion_pair_energy(
        c, torch.zeros(0, 2, dtype=torch.long, device="cuda"),
        torch.zeros(0, dtype=DT, device="cuda"), torch.zeros(0, dtype=DT, device="cuda"), use_customized_ops=True)
    assert e.shape == (0,)
