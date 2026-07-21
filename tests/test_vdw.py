"""Tests for :mod:`torchff.vdw` custom ops vs PyTorch reference paths."""

import pytest
import torch
torch.set_printoptions(precision=8)

from torchff.test_utils import check_op
from torchff.vdw import (
    Vdw,
    compute_vdw_taper,
    compute_vdw_taper_deriv,
    _VDW_TAPER_FACTOR,
)


def test_vdw_custom_requires_sum_output() -> None:
    """Custom vdW ops cannot return per-pair energies; sum_output must stay True."""
    with pytest.raises(ValueError, match="sum_output must be True"):
        Vdw(use_customized_ops=True, sum_output=False)


def test_vdw_lj_taper_requires_switching_distance() -> None:
    """LennardJones taper requires an explicit switching_distance."""
    with pytest.raises(ValueError, match="switching_distance is required"):
        Vdw(function="LennardJones", use_taper=True)


def test_vdw_taper_boundaries() -> None:
    """OpenMM quintic taper has correct values at region boundaries."""
    r_on, r_off = 1.5, 2.0
    r = torch.tensor([1.0, 1.5, 1.75, 2.0, 2.5], dtype=torch.float64)
    taper = compute_vdw_taper(r, r_on, r_off)
    assert taper[0].item() == pytest.approx(1.0)
    assert taper[1].item() == pytest.approx(1.0)
    assert taper[2].item() == pytest.approx(0.5)
    assert taper[3].item() == pytest.approx(0.0)
    assert taper[4].item() == pytest.approx(0.0)


def test_vdw_taper_matches_openmm_formula() -> None:
    """Taper matches the normalized OpenMM polynomial form."""
    r_on, r_off = 1.5, 2.0
    r = torch.linspace(r_on, r_off, 11, dtype=torch.float64)
    taper = compute_vdw_taper(r, r_on, r_off)
    t = (r - r_on) / (r_off - r_on)
    expected = 1.0 + t**3 * (-10.0 + t * (15.0 - 6.0 * t))
    assert torch.allclose(taper, expected, atol=1e-14, rtol=0.0)


def test_vdw_taper_deriv_numeric() -> None:
    """Analytic dS/dr matches central finite differences."""
    r_on, r_off = 1.5, 2.0
    r = torch.tensor([1.6, 1.75, 1.9], dtype=torch.float64, requires_grad=True)
    taper = compute_vdw_taper(r, r_on, r_off)
    (grad,) = torch.autograd.grad(taper.sum(), r)
    expected = compute_vdw_taper_deriv(r.detach(), r_on, r_off)
    assert torch.allclose(grad, expected, atol=1e-10, rtol=0.0)


def _build_all_pairs(num_atoms: int, device: torch.device) -> torch.Tensor:
    """All unique pairs (i < j), shape (P, 2) with P = n*(n-1)/2."""
    idx = torch.arange(num_atoms, device=device, dtype=torch.int64)
    return torch.combinations(idx, r=2)


def _make_synthetic_vdw_system(
    num_atoms: int,
    device: torch.device,
    dtype: torch.dtype,
    box_edge_nm: float = 5.0,
):
    """
    Periodic cubic box of edge ``box_edge_nm`` with ``num_atoms`` coordinates drawn
    uniformly in ``[0, box_edge_nm)`` per axis, per-pair sigma/epsilon, and all
    unordered pairs.
    """
    assert num_atoms == 100
    g = torch.Generator(device=device)
    g.manual_seed(42)
    coords = (
        torch.rand(num_atoms, 3, generator=g, device=device, dtype=dtype) * box_edge_nm
    )
    coords.requires_grad_(True)

    box = torch.diag(
        torch.tensor([box_edge_nm, box_edge_nm, box_edge_nm], device=device, dtype=dtype)
    )
    box.requires_grad_(True)

    pairs = _build_all_pairs(num_atoms, device)
    p = pairs.shape[0]
    assert p == num_atoms * (num_atoms - 1) // 2

    sigma = torch.full((p,), 0.32, device=device, dtype=dtype, requires_grad=True)
    epsilon = torch.full((p,), 0.85, device=device, dtype=dtype, requires_grad=True)

    cutoff = 0.5 * box_edge_nm * (3.0**0.5) + 1e-3

    return coords, pairs, box, sigma, epsilon, cutoff


def _make_two_atom_shell_system(
    device: torch.device,
    dtype: torch.dtype,
    vdw_function: str,
    separation: float,
    cutoff: float = 1.0,
):
    """Two atoms in a large cubic box with pair distance ``separation``."""
    box_edge = 10.0
    coords = torch.zeros(2, 3, device=device, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        coords[1, 0] = separation

    box = torch.diag(
        torch.tensor([box_edge, box_edge, box_edge], device=device, dtype=dtype)
    )
    pairs = torch.tensor([[0, 1]], device=device, dtype=torch.int64)
    sigma = torch.tensor([0.32], device=device, dtype=dtype, requires_grad=True)
    epsilon = torch.tensor([0.85], device=device, dtype=dtype, requires_grad=True)

    if vdw_function == "LennardJones":
        switching_distance = 0.85 * cutoff
    else:
        switching_distance = None

    r_on = switching_distance if vdw_function == "LennardJones" else _VDW_TAPER_FACTOR * cutoff
    assert r_on < separation <= cutoff, (
        f"separation {separation} must lie in taper shell ({r_on}, {cutoff}]"
    )

    return coords, pairs, box, sigma, epsilon, cutoff, switching_distance


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_custom_matches_reference(vdw_function: str, dtype: torch.dtype) -> None:
    """Custom CUDA vdW total energy and gradients match the PyTorch reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    n = 100
    coords, pairs, box, sigma, epsilon, cutoff = _make_synthetic_vdw_system(
        n, device, dtype
    )

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "sigma": sigma,
        "epsilon": epsilon,
        "cutoff": cutoff,
    }

    func = Vdw(
        function=vdw_function,
        use_customized_ops=True,
    ).to(device=device, dtype=dtype)

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=True
    )


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_taper_custom_matches_reference(vdw_function: str, dtype: torch.dtype) -> None:
    """Tapered custom CUDA vdW matches the PyTorch reference path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    coords, pairs, box, sigma, epsilon, cutoff = _make_synthetic_vdw_system(
        100, device, dtype
    )

    switching_distance = 0.85 * cutoff if vdw_function == "LennardJones" else None
    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "sigma": sigma,
        "epsilon": epsilon,
        "cutoff": cutoff,
    }

    func = Vdw(
        function=vdw_function,
        use_customized_ops=True,
        use_taper=True,
        switching_distance=switching_distance,
    ).to(device=device, dtype=dtype)

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        use_taper=True,
        switching_distance=switching_distance,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=True,
    )


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_taper_two_atom_shell(vdw_function: str, dtype: torch.dtype) -> None:
    """Two-atom system inside the taper shell matches between CUDA and reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    cutoff = 1.0
    r_on = 0.85 * cutoff if vdw_function == "LennardJones" else _VDW_TAPER_FACTOR * cutoff
    separation = 0.5 * (r_on + cutoff)

    coords, pairs, box, sigma, epsilon, cutoff, switching_distance = (
        _make_two_atom_shell_system(device, dtype, vdw_function, separation, cutoff)
    )

    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "sigma": sigma,
        "epsilon": epsilon,
        "cutoff": cutoff,
    }

    func = Vdw(
        function=vdw_function,
        use_customized_ops=True,
        use_taper=True,
        switching_distance=switching_distance,
    ).to(device=device, dtype=dtype)

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        use_taper=True,
        switching_distance=switching_distance,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
        verbose=True,
    )


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_taper_grad_vs_autograd(vdw_function: str, dtype: torch.dtype) -> None:
    """CUDA coord gradients match autograd on the tapered Python reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    cutoff = 1.0
    r_on = 0.85 * cutoff if vdw_function == "LennardJones" else _VDW_TAPER_FACTOR * cutoff
    separation = 0.5 * (r_on + cutoff)

    coords, pairs, box, sigma, epsilon, cutoff, switching_distance = (
        _make_two_atom_shell_system(device, dtype, vdw_function, separation, cutoff)
    )

    func_cuda = Vdw(
        function=vdw_function,
        use_customized_ops=True,
        use_taper=True,
        switching_distance=switching_distance,
    ).to(device=device, dtype=dtype)

    ene_cuda = func_cuda(coords, pairs, box, sigma, epsilon, cutoff)
    ene_cuda.backward()
    grad_cuda = coords.grad.clone()
    coords.grad = None

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        use_taper=True,
        switching_distance=switching_distance,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)
    ene_py = func_ref(coords, pairs, box, sigma, epsilon, cutoff)
    ene_py.backward()
    grad_py = coords.grad.clone()

    assert torch.allclose(ene_cuda, ene_py, atol=1e-6, rtol=0.0)
    assert torch.allclose(grad_cuda, grad_py, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("vdw_function", ["LennardJones", "AmoebaVdw147"])
@pytest.mark.parametrize("dtype", [torch.float64])
def test_vdw_taper_disabled_matches_hard_cutoff(vdw_function: str, dtype: torch.dtype) -> None:
    """use_taper=False reproduces the hard-cutoff behavior."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for customized vdW ops.")

    device = torch.device("cuda")
    coords, pairs, box, sigma, epsilon, cutoff = _make_synthetic_vdw_system(
        100, device, dtype
    )
    kwargs = {
        "coords": coords,
        "pairs": pairs,
        "box": box,
        "sigma": sigma,
        "epsilon": epsilon,
        "cutoff": cutoff,
    }

    func = Vdw(
        function=vdw_function,
        use_customized_ops=True,
        use_taper=False,
    ).to(device=device, dtype=dtype)

    func_ref = Vdw(
        function=vdw_function,
        use_customized_ops=False,
        use_taper=False,
        cuda_graph_compat=True,
    ).to(device=device, dtype=dtype)

    check_op(
        func,
        func_ref,
        kwargs,
        check_grad=True,
        atol=1e-6,
        rtol=0.0,
    )
