"""torchff.slaterelec: Slater-penetrated multipole electrostatics, energy + field (matvec).

1. reference self-consistency (field == autograd of the energy), any device;
2. the device math (csrc/slaterelec/slater_elec.cuh on double and on nested Dual<double>)
   against torch double backward, via a host g++ build -- no CUDA needed;
3. CUDA kernels against the references: energies, first-order gradients of the energy, the
   field, and the field's VJP with respect to every input.
"""

import os
import shutil
import subprocess

import pytest
import torch

from torchff import slaterelec as se

DT = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(os.path.dirname(HERE), "csrc")


def _system(n=9, K=10, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = (torch.rand(n, 3, generator=g, dtype=DT) * 4.0).to(device).requires_grad_(True)
    ii, jj = torch.triu_indices(n, n, 1)
    pairs = torch.stack([ii, jj], 1).to(device)
    b = (torch.rand(n, generator=g, dtype=DT) + 1.5).to(device).requires_grad_(True)
    gate = torch.rand(pairs.shape[0], generator=g, dtype=DT).to(device).requires_grad_(True)
    m = (torch.randn(n, K, generator=g, dtype=DT) * 0.3).to(device).requires_grad_(True)
    n_ = torch.zeros(n, K, dtype=DT)
    n_[:, 0] = torch.randint(1, 8, (n,), generator=g).to(DT)
    n_ = n_.to(device).requires_grad_(True)
    return coords, pairs, b, gate, m, n_


# --- 1 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 4, 10])
def test_field_ref_is_the_gradient_of_the_energy_ref(K):
    coords, pairs, b, gate, m, n = _system(K=K)
    e = se.slater_elec_pair_energy_ref(coords, pairs, b, gate, m, n)
    (f_auto,) = torch.autograd.grad(e.sum(), m)
    f = se.slater_elec_field_ref(coords, pairs, b, gate, m, n)
    assert torch.allclose(f, f_auto, rtol=1e-12, atol=1e-14)


def test_lower_rank_is_higher_rank_with_zeros():
    coords, pairs, b, gate, m, n = _system(K=10)
    m4 = m.detach().clone(); m4[:, 4:] = 0
    e10 = se.slater_elec_pair_energy_ref(coords, pairs, b, gate, m4, n.detach())
    e4 = se.slater_elec_pair_energy_ref(coords, pairs, b, gate, m4[:, :4], n.detach()[:, :4])
    assert torch.allclose(e10, e4, rtol=1e-12, atol=1e-15)


# --- 2 ---------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_device_math_matches_torch(tmp_path):
    exe = tmp_path / "check_slater_elec"
    subprocess.run(["g++", "-std=c++17", "-O1", "-I", os.path.join(HERE, "host"), "-I", CSRC,
                    os.path.join(HERE, "host", "check_slater_elec.cpp"), "-o", str(exe)], check=True)
    out = {}
    for line in subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.splitlines():
        p = line.split()
        if p[0] == "e":
            out["e"] = float(p[1])
        else:
            out.setdefault(p[0], {})[int(p[1])] = float(p[2])

    m_i = torch.tensor([0.31, 0.12, -0.05, 0.08, 0.021, -0.013, 0.007, -0.011, 0.004, -0.010], dtype=DT)
    m_j = torch.tensor([-0.42, -0.07, 0.11, 0.02, -0.015, 0.009, 0.012, 0.006, -0.008, 0.009], dtype=DT)
    n_i = torch.tensor([1.0] + [0.0] * 9, dtype=DT); n_j = torch.tensor([6.0] + [0.0] * 9, dtype=DT)
    dr = torch.tensor([2.3, -1.1, 0.7], dtype=DT); b = torch.tensor([1.9, 2.4], dtype=DT)
    lam = torch.tensor([[0.7, -0.2, 0.4, 0.1, 0.05, -0.03, 0.02, 0.06, -0.01, 0.03],
                        [-0.3, 0.5, 0.1, -0.6, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02]], dtype=DT)
    leaves = [t.clone().requires_grad_(True) for t in (m_i, m_j, n_i, n_j, dr, b)]
    mi, mj, ni, nj, d, bb = leaves
    coords = torch.stack([torch.zeros(3, dtype=DT), d])
    m = torch.stack([mi, mj]); n = torch.stack([ni, nj])
    e = se.slater_elec_pair_energy_ref(coords, torch.tensor([[0, 1]]), bb, torch.ones(1, dtype=DT), m, n)[0]
    assert abs(e.item() - out["e"]) < 1e-14

    g = torch.autograd.grad(e, leaves, create_graph=True)
    for name, gt in zip(["g_mi", "g_mj", "g_ni", "g_nj", "gdr"], g[:5]):
        for k in range(gt.numel()):
            assert abs(gt[k].item() - out[name][k]) < 1e-12 * max(1.0, abs(gt[k].item())), (name, k)
    assert abs(g[5][0].item() - out["g_bi"][0]) < 1e-13 and abs(g[5][1].item() - out["g_bj"][0]) < 1e-13
    for name, gt in (("f_i", g[0]), ("f_j", g[1])):
        for k in range(10):
            assert abs(gt[k].item() - out[name][k]) < 1e-13

    # field VJP: d/d(everything) of  lam . dE/dm   (nested duals on the device side)
    L = (torch.stack([g[0], g[1]]) * lam).sum()
    h = torch.autograd.grad(L, leaves, retain_graph=True)
    assert abs(L.item() - out["vjp_e"][0]) < 1e-14
    for name, ht in zip(["vjp_mi", "vjp_mj", "vjp_ni", "vjp_nj", "vjp_dr"], h[:5]):
        for k in range(ht.numel()):
            assert abs(ht[k].item() - out[name][k]) < 1e-12 * max(1.0, abs(ht[k].item())), (name, k)
    assert abs(h[5][0].item() - out["vjp_bi"][0]) < 1e-13 and abs(h[5][1].item() - out["vjp_bj"][0]) < 1e-13

    # energy HVP: every input seeded with a direction v; tangent(dE/d*) must be H v, i.e.
    # d/d(everything) of  v . dE/d(everything)  -- the double backward of the energy op
    v = [torch.tensor(t, dtype=DT) for t in (
        [0.7, -0.2, 0.4, 0.1, 0.05, -0.03, 0.02, 0.06, -0.01, 0.03],
        [-0.3, 0.5, 0.1, -0.6, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02],
        [0.2] + [0.0] * 9, [-0.4] + [0.0] * 9, [0.13, -0.27, 0.05], [0.31, -0.17])]
    Lv = sum((gt * vt).sum() for gt, vt in zip(g, v))
    hv = torch.autograd.grad(Lv, leaves)
    assert abs(Lv.item() - out["hvp_e"][0]) < 1e-13 * max(1.0, abs(Lv.item()))
    for name, ht in zip(["hvp_mi", "hvp_mj", "hvp_ni", "hvp_nj", "hvp_dr"], hv[:5]):
        for k in range(ht.numel()):
            assert abs(ht[k].item() - out[name][k]) < 1e-11 * max(1.0, abs(ht[k].item())), (name, k)
    assert abs(hv[5][0].item() - out["hvp_bi"][0]) < 1e-12 * max(1.0, abs(hv[5][0].item()))
    assert abs(hv[5][1].item() - out["hvp_bj"][0]) < 1e-12 * max(1.0, abs(hv[5][1].item()))

    # Pauli: the same pair as per-pair polytensors with one combined exponent
    pl = [t.clone().requires_grad_(True) for t in (m_i, m_j, dr)]
    pb = torch.tensor([2.1], dtype=DT, requires_grad=True)
    pcoords = torch.stack([torch.zeros(3, dtype=DT), pl[2]])
    ep = se.slater_pauli_pair_energy_ref(pcoords, torch.tensor([[0, 1]]), pb, pl[0][None], pl[1][None])[0]
    assert abs(ep.item() - out["pauli_e"][0]) < 1e-13 * max(1.0, abs(ep.item()))
    gp = torch.autograd.grad(ep, pl + [pb], create_graph=True)
    for name, gt in zip(["pauli_gai", "pauli_gaj", "pauli_gdr", "pauli_gb"], gp):
        for k in range(gt.numel()):
            assert abs(gt[k].item() - out[name][k]) < 1e-12 * max(1.0, abs(gt[k].item())), (name, k)
    vp = [torch.tensor(t, dtype=DT) for t in (
        [0.7, -0.2, 0.4, 0.1, 0.05, -0.03, 0.02, 0.06, -0.01, 0.03],
        [-0.3, 0.5, 0.1, -0.6, 0.02, 0.04, -0.05, 0.01, 0.03, -0.02], [0.13, -0.27, 0.05], [0.31])]
    Lp = sum((gt * vt).sum() for gt, vt in zip(gp, vp))
    hp = torch.autograd.grad(Lp, pl + [pb])
    assert abs(Lp.item() - out["pauli_hvp_e"][0]) < 1e-13 * max(1.0, abs(Lp.item()))
    for name, ht in zip(["pauli_hvp_ai", "pauli_hvp_aj", "pauli_hvp_dr", "pauli_hvp_b"], hp):
        for k in range(ht.numel()):
            assert abs(ht[k].item() - out[name][k]) < 1e-11 * max(1.0, abs(ht[k].item())), (name, k)


# --- 3 ---------------------------------------------------------------------------------------

needs_cuda = pytest.mark.skipif(
    not (torch.cuda.is_available() and se.HAVE_KERNELS), reason="needs CUDA + torchff_slaterelec")


@needs_cuda
@pytest.mark.parametrize("K", [1, 4, 10])
def test_energy_kernel_matches_ref_first_order(K):
    args = _system(K=K, device="cuda")
    coords, pairs, b, gate, m, n = args
    leaves = (coords, b, gate, m, n)
    e_ref = se.slater_elec_pair_energy_ref(*args)
    e_op = se.slater_elec_pair_energy(*args, use_customized_ops=True)
    assert torch.allclose(e_op, e_ref, rtol=1e-12, atol=1e-15)
    w = torch.randn_like(e_ref)
    g_ref = torch.autograd.grad((e_ref * w).sum(), leaves)
    g_op = torch.autograd.grad((e_op * w).sum(), leaves)
    for a, c in zip(g_ref, g_op):
        assert torch.allclose(a, c, rtol=1e-10, atol=1e-13)


@needs_cuda
@pytest.mark.parametrize("K", [1, 4, 10])
def test_energy_kernel_double_backward_matches_ref(K):
    """A force loss through the kernel: ``|| dE/dcoords ||^2`` (+ the other first derivatives,
    weighted) backpropagated into every input, kernel vs reference, and gradgradcheck."""
    args = _system(K=K, device="cuda")
    coords, pairs, b, gate, m, n = args
    leaves = (coords, b, gate, m, n)
    ws = [torch.randn_like(t) for t in leaves]

    def loss(fn):
        e = fn(*args)
        w = torch.linspace(0.5, 1.5, e.numel(), dtype=e.dtype, device=e.device)
        g = torch.autograd.grad((e * w).sum(), leaves, create_graph=True)
        return sum((gk * gk * wk).sum() + (gk * wk).sum() for gk, wk in zip(g, ws))

    l_ref = loss(se.slater_elec_pair_energy_ref)
    l_op = loss(lambda *a: se.slater_elec_pair_energy(*a, use_customized_ops=True))
    assert torch.allclose(l_op, l_ref, rtol=1e-10)
    h_ref = torch.autograd.grad(l_ref, leaves)
    h_op = torch.autograd.grad(l_op, leaves)
    for a, c in zip(h_ref, h_op):
        assert torch.allclose(a, c, rtol=1e-9, atol=1e-12)
    assert torch.autograd.gradgradcheck(
        lambda c, bb, gg, mm, nn: se.slater_elec_pair_energy(c, pairs, bb, gg, mm, nn, use_customized_ops=True),
        leaves, nondet_tol=1e-11)


@needs_cuda
@pytest.mark.parametrize("K", [1, 4, 10])
def test_field_kernel_and_its_vjp_match_ref(K):
    args = _system(K=K, device="cuda")
    coords, pairs, b, gate, m, n = args
    leaves = (coords, b, gate, m, n)
    f_ref = se.slater_elec_field_ref(*args)
    f_op = se.slater_elec_field(*args, use_customized_ops=True)
    assert f_op.shape == (m.shape[0], K)
    assert torch.allclose(f_op, f_ref, rtol=1e-12, atol=1e-14)
    lam = torch.randn_like(f_ref)
    v_ref = torch.autograd.grad((f_ref * lam).sum(), leaves)
    v_op = torch.autograd.grad((f_op * lam).sum(), leaves)
    for a, c in zip(v_ref, v_op):
        assert torch.allclose(a, c, rtol=1e-10, atol=1e-13)
    assert torch.autograd.gradcheck(
        lambda c, bb, gg, mm, nn: se.slater_elec_field(c, pairs, bb, gg, mm, nn, use_customized_ops=True),
        leaves, nondet_tol=1e-12)


def _pauli_system(n_pairs=30, K=10, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    n = 12
    coords = (torch.rand(n, 3, generator=g, dtype=DT) * 4.0).to(device).requires_grad_(True)
    ii, jj = torch.triu_indices(n, n, 1)
    sel = torch.randperm(ii.numel(), generator=g)[:n_pairs]
    pairs = torch.stack([ii[sel], jj[sel]], 1).to(device)
    b = (torch.rand(n_pairs, generator=g, dtype=DT) + 1.5).to(device).requires_grad_(True)
    a_i = (torch.randn(n_pairs, K, generator=g, dtype=DT) * 0.3).to(device).requires_grad_(True)
    a_j = (torch.randn(n_pairs, K, generator=g, dtype=DT) * 0.3).to(device).requires_grad_(True)
    return coords, pairs, b, a_i, a_j


@needs_cuda
@pytest.mark.parametrize("K", [1, 4, 10])
def test_pauli_kernel_matches_ref_to_second_order(K):
    args = _pauli_system(K=K, device="cuda")
    coords, pairs, b, a_i, a_j = args
    leaves = (coords, b, a_i, a_j)
    ws = [torch.randn_like(t) for t in leaves]

    def run(fn):
        e = fn(*args)
        w = torch.linspace(0.5, 1.5, e.numel(), dtype=e.dtype, device=e.device)
        g = torch.autograd.grad((e * w).sum(), leaves, create_graph=True)
        loss = sum((gk * gk * wk).sum() + (gk * wk).sum() for gk, wk in zip(g, ws))
        h = torch.autograd.grad(loss, leaves)
        return e.detach(), [x.detach() for x in g], [x.detach() for x in h]

    e_r, g_r, h_r = run(se.slater_pauli_pair_energy_ref)
    e_o, g_o, h_o = run(lambda *a: se.slater_pauli_pair_energy(*a, use_customized_ops=True))
    assert torch.allclose(e_o, e_r, rtol=1e-12, atol=1e-15)
    for x, y in zip(g_r, g_o):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-13)
    for x, y in zip(h_r, h_o):
        assert torch.allclose(x, y, rtol=1e-9, atol=1e-12)
    assert torch.autograd.gradgradcheck(
        lambda c, bb, ai, aj: se.slater_pauli_pair_energy(c, pairs, bb, ai, aj, use_customized_ops=True),
        leaves, nondet_tol=1e-11)
