"""Van der Waals (vdW) pair energies: Lennard-Jones 12-6 and AMOEBA buffered 14-7."""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torchff_vdw
from .pbc import PBC

_VDW_TAPER_FACTOR = 0.9  # AmoebaVdwForce default (OpenMM fixed inner distance)


def vdw_taper_coefficients(r_on: float, r_off: float) -> tuple[float, float, float]:
    """
    Return OpenMM quintic taper coefficients (C3, C4, C5) for the x-form switch.

    With ``width = r_on - r_off`` (negative), ``S(r) = 1 + x^3 (C3 + x(C4 + x C5))``
    where ``x = r - r_on``.
    """
    width = r_on - r_off
    return 10.0 / width**3, 15.0 / width**4, 6.0 / width**5


def compute_vdw_taper(
    r: torch.Tensor, r_on: float, r_off: float
) -> torch.Tensor:
    """
    OpenMM-compatible quintic taper S(r).

    Returns 1 for ``r <= r_on``, 0 for ``r >= r_off``, and a smooth quintic
    between ``r_on`` and ``r_off``.
    """
    c3, c4, c5 = vdw_taper_coefficients(r_on, r_off)
    delta = r - r_on
    taper_shell = 1.0 + delta * delta * delta * (c3 + delta * (c4 + delta * c5))
    return torch.where(
        r <= r_on,
        torch.ones_like(r),
        torch.where(r >= r_off, torch.zeros_like(r), taper_shell),
    )


def compute_vdw_taper_deriv(
    r: torch.Tensor, r_on: float, r_off: float
) -> torch.Tensor:
    """Derivative dS/dr of the OpenMM quintic taper; zero outside the taper shell."""
    c3, c4, c5 = vdw_taper_coefficients(r_on, r_off)
    delta = r - r_on
    dtaper = delta * delta * (3.0 * c3 + delta * (4.0 * c4 + delta * 5.0 * c5))
    return torch.where(
        (r <= r_on) | (r >= r_off),
        torch.zeros_like(r),
        dtaper,
    )


def _resolve_vdw_taper_on(
    function: str,
    cutoff: float,
    use_taper: bool,
    switching_distance: float | None,
) -> float | None:
    if not use_taper:
        return None
    if function == "AmoebaVdw147":
        return _VDW_TAPER_FACTOR * cutoff
    if switching_distance is None:
        raise ValueError(
            "switching_distance is required when use_taper=True for LennardJones"
        )
    if not (0.0 < switching_distance < cutoff):
        raise ValueError(
            f"switching_distance must satisfy 0 < switching_distance < cutoff "
            f"(got switching_distance={switching_distance}, cutoff={cutoff})"
        )
    return switching_distance


@torch._dynamo.disable
def compute_vdw_14_7_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
    r_on: float | None = None,
) -> torch.Tensor:
    """
    Compute AMOEBA buffered 14-7 vdW pair energies via custom CUDA/C++ ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    pairs : torch.Tensor
        Shape (P, 2), integer indices (i, j) of interacting pairs.
    box : torch.Tensor
        Shape (3, 3) or broadcastable, periodic box (same convention as :mod:`torchff.pbc`).
    sigma : torch.Tensor
        Per-pair or type-pair :math:`\\sigma` (see ``atom_types``).
    epsilon : torch.Tensor
        Per-pair or type-pair :math:`\\epsilon` (see ``atom_types``).
    cutoff : float
        Distance cutoff; interactions beyond cutoff are excluded by the kernel.
    atom_types : torch.Tensor, optional
        If provided, used by the backend for type-based indexing together with ``sigma`` and ``epsilon``.
    r_on : float, optional
        Inner distance where tapering begins (``r_off`` is ``cutoff``). If None, no taper is applied.

    Returns
    -------
    torch.Tensor
        Scalar total vdW energy for the buffered 14-7 potential.
    """
    return torch.ops.torchff.compute_vdw_14_7_energy(
        coords, pairs, box, sigma, epsilon, cutoff, atom_types, r_on if r_on is not None else -1.0
    )


@torch._dynamo.disable
def compute_lennard_jones_energy(
    coords: torch.Tensor,
    pairs: torch.Tensor,
    box: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
    cutoff: float,
    atom_types: torch.Tensor | None = None,
    r_on: float | None = None,
) -> torch.Tensor:
    """
    Compute Lennard-Jones 12-6 vdW pair energies via custom CUDA/C++ ops.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (N, 3), atom coordinates.
    pairs : torch.Tensor
        Shape (P, 2), integer indices (i, j) of interacting pairs.
    box : torch.Tensor
        Shape (3, 3) or broadcastable, periodic box (same convention as :mod:`torchff.pbc`).
    sigma : torch.Tensor
        Per-pair or type-pair :math:`\\sigma` (see ``atom_types``).
    epsilon : torch.Tensor
        Per-pair or type-pair :math:`\\epsilon` (see ``atom_types``).
    cutoff : float
        Distance cutoff; interactions beyond cutoff are excluded by the kernel.
    atom_types : torch.Tensor, optional
        If provided, used by the backend for type-based indexing together with ``sigma`` and ``epsilon``.
    r_on : float, optional
        Inner distance where tapering begins (``r_off`` is ``cutoff``). If None, no taper is applied.

    Returns
    -------
    torch.Tensor
        Scalar total Lennard-Jones energy.
    """
    return torch.ops.torchff.compute_lennard_jones_energy(
        coords, pairs, box, sigma, epsilon, cutoff, atom_types, r_on if r_on is not None else -1.0
    )


def compute_lennard_jones_energy_ref(
    r_ij,
    sigma_ij,
    epsilon_ij,
    sum=True,
    *,
    r_on: float | None = None,
    r_off: float | None = None,
):
    """
    Reference Lennard-Jones 12-6 pair energy in PyTorch.

    Per pair:
    :math:`E_{ij} = 4 \\epsilon_{ij} \\left[ (\\sigma_{ij}/r_{ij})^{12} - (\\sigma_{ij}/r_{ij})^6 \\right]`.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    sigma_ij : torch.Tensor
        :math:`\\sigma` for each pair, same shape as ``r_ij`` (after broadcast).
    epsilon_ij : torch.Tensor
        :math:`\\epsilon` for each pair, same shape as ``r_ij`` (after broadcast).
    sum : bool, optional
        If True (default), return the sum over pairs; otherwise return per-pair energies.
    r_on : float, optional
        Inner distance where OpenMM quintic taper begins. If None, no taper is applied.
    r_off : float, optional
        Outer distance where taper reaches zero (typically the cutoff). Required when ``r_on`` is set.

    Returns
    -------
    torch.Tensor
        Scalar total energy if ``sum`` is True, else shape (P,) per-pair energies.
    """
    tmp = (sigma_ij / r_ij) ** 6
    ene_ij = 4 * epsilon_ij * tmp * (tmp - 1)
    if r_on is not None:
        if r_off is None:
            raise ValueError("r_off is required when r_on is set")
        ene_ij = ene_ij * compute_vdw_taper(r_ij, r_on, r_off)
    return torch.sum(ene_ij) if sum else ene_ij


def compute_vdw_14_7_energy_ref(
    r_ij,
    sigma_ij,
    epsilon_ij,
    sum=True,
    *,
    r_on: float | None = None,
    r_off: float | None = None,
):
    """
    Reference AMOEBA buffered 14-7 vdW pair energy in PyTorch.

    With :math:`\\rho = r_{ij} / \\sigma_{ij}`,
    :math:`E_{ij} = \\epsilon_{ij} \\left( \\frac{1.07}{\\rho + 0.07} \\right)^7 \\left( \\frac{1.12}{\\rho^7 + 0.12} - 2 \\right)`.

    Parameters
    ----------
    r_ij : torch.Tensor
        Pair distances, shape (P,) or broadcastable.
    sigma_ij : torch.Tensor
        :math:`\\sigma` for each pair, same shape as ``r_ij`` (after broadcast).
    epsilon_ij : torch.Tensor
        :math:`\\epsilon` for each pair, same shape as ``r_ij`` (after broadcast).
    sum : bool, optional
        If True (default), return the sum over pairs; otherwise return per-pair energies.
    r_on : float, optional
        Inner distance where OpenMM quintic taper begins. If None, no taper is applied.
    r_off : float, optional
        Outer distance where taper reaches zero (typically the cutoff). Required when ``r_on`` is set.

    Returns
    -------
    torch.Tensor
        Scalar total energy if ``sum`` is True, else shape (P,) per-pair energies.
    """
    rho = r_ij / sigma_ij
    ene_ij = epsilon_ij * (1.07 / (rho + 0.07)) ** 7 * (1.12 / (rho**7 + 0.12) - 2.0)
    if r_on is not None:
        if r_off is None:
            raise ValueError("r_off is required when r_on is set")
        ene_ij = ene_ij * compute_vdw_taper(r_ij, r_on, r_off)
    return torch.sum(ene_ij) if sum else ene_ij


class Vdw(nn.Module):
    """
    Van der Waals pair energy module (Lennard-Jones 12-6 or AMOEBA buffered 14-7).

    Dispatches to :func:`compute_lennard_jones_energy` / :func:`compute_vdw_14_7_energy`
    when :attr:`use_customized_ops` is True; otherwise uses minimum-image displacements
    via :class:`torchff.pbc.PBC` and the reference formulas
    :func:`compute_lennard_jones_energy_ref` / :func:`compute_vdw_14_7_energy_ref`.
    """

    def __init__(
        self,
        function: Literal['LennardJones', 'AmoebaVdw147'] = 'LennardJones',
        cutoff: Optional[float] = None,
        use_customized_ops: bool = False,
        use_type_pairs: bool = False,
        sum_output: bool = True,
        cuda_graph_compat: bool = True,
        use_taper: bool = False,
        switching_distance: float | None = None,
    ):
        """
        Parameters
        ----------
        function : {'LennardJones', 'AmoebaVdw147'}, optional
            Potential form: standard LJ 12-6 or AMOEBA buffered 14-7.
        cutoff : float, optional
            Stored on the module; the active cutoff is the ``cutoff`` argument to :meth:`forward`.
        use_customized_ops : bool, optional
            If True, use custom CUDA/C++ kernels; otherwise use the PyTorch reference path.
        use_type_pairs : bool, optional
            If True, ``sigma`` and ``epsilon`` are indexed by ``atom_types`` for each pair
            (shape ``(n_types, n_types)``).
        sum_output : bool, optional
            If True (default), return a scalar sum over pairs. Must be True when
            ``use_customized_ops`` is True because the custom kernels only return total energy.
            When ``use_customized_ops`` is False, if False return per-pair energies of shape ``(P,)``.
        cuda_graph_compat : bool, optional
            If True (default), apply the cutoff with :func:`torch.where` so tensor shapes are
            stable; if False, distances are filtered with boolean indexing before the energy expression.
        use_taper : bool, optional
            If True, apply the OpenMM quintic multiplicative taper between ``r_on`` and ``cutoff``.
            For LennardJones, ``switching_distance`` must be set; for AmoebaVdw147, ``r_on`` is
            ``0.9 * cutoff`` (OpenMM ``AmoebaVdwForce`` default).
        switching_distance : float, optional
            Inner taper distance for LennardJones (OpenMM ``NonbondedForce.switchingDistance``).
            Ignored for AmoebaVdw147.
        """
        super().__init__()
        self.use_customized_ops = use_customized_ops
        self.use_type_pairs = use_type_pairs
        self.sum_output = sum_output
        self.cuda_graph_compat = cuda_graph_compat
        self.use_taper = use_taper
        self.switching_distance = switching_distance
        self.pbc = PBC()
        self.cutoff = cutoff
        if self.use_customized_ops and not self.sum_output:
            raise ValueError(
                "sum_output must be True when use_customized_ops is True "
                "(custom vdW kernels only compute total energy, not per-pair terms)."
            )
        self.function = function
        assert self.function in ('LennardJones', 'AmoebaVdw147'), f'Invalid vdw function: {function}'
        if self.use_taper and self.function == 'LennardJones' and self.switching_distance is None:
            raise ValueError(
                "switching_distance is required when use_taper=True for LennardJones"
            )
    
    def expand_type_pairs(self, sigma, epsilon, pairs, atom_types):
        if self.use_type_pairs:
            atypes_i, atypes_j = atom_types[pairs[:, 0]], atom_types[pairs[:, 1]]
            sigma_ij = sigma[atypes_i, atypes_j]
            epsilon_ij = epsilon[atypes_i, atypes_j]
            return sigma_ij, epsilon_ij
        else:
            return sigma, epsilon

    def forward(
        self,
        coords: torch.Tensor,
        pairs: torch.Tensor,
        box: torch.Tensor,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        cutoff: float,
        atom_types: torch.Tensor | None = None,
    ):
        """
        Compute vdW energy for the configured potential.

        Parameters
        ----------
        coords : torch.Tensor
            Shape (N, 3), atom coordinates.
        pairs : torch.Tensor
            Shape (P, 2), pair indices (i, j).
        box : torch.Tensor
            Periodic box, same convention as :class:`torchff.pbc.PBC`.
        sigma : torch.Tensor
            Per-pair ``(P,)`` or type table ``(T, T)`` when :attr:`use_type_pairs` is True.
        epsilon : torch.Tensor
            Same layout as ``sigma``.
        cutoff : float
            Pair distance cutoff.
        atom_types : torch.Tensor, optional
            Shape (N,), integer atom types; required when :attr:`use_type_pairs` is True.

        Returns
        -------
        torch.Tensor
            If :attr:`use_customized_ops` is True, scalar total energy from the custom op.
            Otherwise per-pair energies of shape (P,), or a scalar if :attr:`sum_output` is True.
        """
        r_on = _resolve_vdw_taper_on(
            self.function, cutoff, self.use_taper, self.switching_distance
        )
        if self.use_customized_ops:
            if self.function == 'LennardJones':
                return compute_lennard_jones_energy(
                    coords, pairs, box, sigma, epsilon, cutoff, atom_types, r_on
                )
            else:
                return compute_vdw_14_7_energy(
                    coords, pairs, box, sigma, epsilon, cutoff, atom_types, r_on
                )
        else:
            drVecs = self.pbc(coords[pairs[:, 1]] - coords[pairs[:, 0]], box)
            sigma_ij, epsilon_ij = self.expand_type_pairs(sigma, epsilon, pairs, atom_types)
            dr = torch.norm(drVecs, dim=1)
            taper_kwargs = {"r_on": r_on, "r_off": cutoff} if r_on is not None else {}
            if not self.cuda_graph_compat:
                dr = dr[dr <= cutoff]
            if self.function == 'LennardJones':
                ene_pairs = compute_lennard_jones_energy_ref(
                    dr, sigma_ij, epsilon_ij, sum=False, **taper_kwargs
                )
            else:
                ene_pairs = compute_vdw_14_7_energy_ref(
                    dr, sigma_ij, epsilon_ij, sum=False, **taper_kwargs
                )
            if self.cuda_graph_compat:
                ene_pairs = torch.where(dr <= cutoff, ene_pairs, 0.0)
            if self.sum_output:
                return torch.sum(ene_pairs)
            else:
                return ene_pairs
