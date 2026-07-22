"""
OpenMM production MD for a periodic TIP3P water box.

Loads ``examples/water_300.pdb`` and runs:

1. Energy minimization
2. NVT equilibration at 298.15 K (1 fs timestep, 500 ps)
3. NPT equilibration with a Monte Carlo barostat (1 fs, 500 ps)
4. NPT production (1 fs, 5 ns)

Trajectories and logs are written to ``examples/tip3p/md_openmm/`` (native OpenMM)
or ``examples/tip3p/md_torchff/`` when :data:`CONFIG` ``use_torchff`` is enabled.

All simulation parameters live in :data:`CONFIG` at the bottom of this file.

Run on a NERSC GPU compute node (interactive or batch)::

    srun --nodes 1 --qos interactive --time 2:00:00 --constraint gpu --gres=gpu:1 \\
        bash -c 'module load conda && mamba activate openmm-torch-py312-cu124 && \\
        python examples/tip3p/md_prod.py --torchff'

Or from ``examples/tip3p``::

    python md_prod.py
    python md_prod.py --torchff
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from md import (
    apply_tip3p_nonbonded_settings,
    assert_close_to_openmm_reference,
    build_tip3p_torchff_config,
    create_tip3p_torchforce,
    openmm_system_with_tip3p_torchforce,
    report_openmm_torchff_comparison,
)
from model import Tip3pTorchFF

if TYPE_CHECKING:
    from openmmtorch import TorchForce


@dataclass(frozen=True)
class TorchFFBundle:
    """Cached TorchFF model shared across MD phases."""

    model: Tip3pTorchFF


def create_torchforce(config: dict, model: Tip3pTorchFF) -> TorchForce:
    """Build a fresh OpenMM-Torch force (one per OpenMM System)."""
    return create_tip3p_torchforce(
        model,
        use_cuda_graphs=config["use_cuda_graphs"],
        cuda_graph_warmup_steps=config["cuda_graph_warmup_steps"],
    )


def ps_to_steps(duration_ps: float, timestep_ps: float) -> int:
    """Convert a duration in ps to an integer number of integration steps."""
    return int(round(duration_ps / timestep_ps))


def resolve_output_dir(config: dict) -> Path:
    """Return the output directory for the selected force-field backend."""
    if config["use_torchff"]:
        return Path(config["output_dir_torchff"]).resolve()
    return Path(config["output_dir_openmm"]).resolve()


def prepare_torchff_bundle(config: dict, pdb_path: Path) -> TorchFFBundle:
    """Build :class:`Tip3pTorchFF` from the input PDB and cutoff."""
    cfg, _, _ = build_tip3p_torchff_config(
        pdb_path,
        config["nonbonded_cutoff_nm"],
        use_switching_function=config["use_switching_function"],
        switching_distance_nm=config["switching_distance_nm"],
    )
    model = Tip3pTorchFF(
        cfg,
        use_customized_ops=config["use_customized_ops"],
        vdw_taper=config["vdw_taper"],
        switching_distance_nm=config["switching_distance_nm"],
    )
    return TorchFFBundle(model=model)


def validate_torchff_vs_openmm(config: dict, pdb_path: Path) -> None:
    """Assert TorchFF energy terms match OpenMM at the input PDB geometry."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TorchFF energy validation.")

    cutoff_nm = config["nonbonded_cutoff_nm"]
    switching_nm = config["switching_distance_nm"]
    cfg, _, openmm_ref = build_tip3p_torchff_config(
        pdb_path,
        cutoff_nm,
        use_switching_function=config["use_switching_function"],
        switching_distance_nm=switching_nm,
    )

    pdb = app.PDBFile(str(pdb_path))
    dtype = torch.float64
    device = torch.device("cuda")
    coords_nm = torch.tensor(
        np.asarray(pdb.getPositions(asNumpy=True)), dtype=dtype, device=device
    )
    box_nm = torch.tensor(
        [v.value_in_unit(unit.nanometer) for v in pdb.topology.getPeriodicBoxVectors()],
        dtype=dtype,
        device=device,
    )

    model = Tip3pTorchFF(
        cfg,
        use_customized_ops=config["use_customized_ops"],
        vdw_taper=config["vdw_taper"],
        switching_distance_nm=switching_nm,
    ).to(device, dtype)
    model.eval()

    print("Validating TorchFF vs OpenMM energies at input PDB geometry...")
    with torch.no_grad():
        assert_close_to_openmm_reference(model, coords_nm, box_nm, openmm_ref)
    print(report_openmm_torchff_comparison(model, coords_nm, box_nm, openmm_ref))
    print("Energy validation passed.")


def build_system(
    topology: app.Topology,
    config: dict,
    *,
    use_barostat: bool,
    torchff_bundle: TorchFFBundle | None = None,
) -> mm.System:
    """Build a TIP3P system (native OpenMM or TorchFF) with optional Monte Carlo barostat."""
    if config["use_torchff"]:
        if torchff_bundle is None:
            raise ValueError("torchff_bundle is required when use_torchff=True")
        torch_force = create_torchforce(config, torchff_bundle.model)
        system = openmm_system_with_tip3p_torchforce(
            torchff_bundle.model,
            torch_force,
            topology,
        )
    else:
        ff = app.ForceField(config["forcefield"])
        system = ff.createSystem(
            topology,
            nonbondedMethod=getattr(app, config["nonbonded_method"]),
            nonbondedCutoff=config["nonbonded_cutoff_nm"] * unit.nanometer,
            constraints=config["constraints"],
            rigidWater=config["rigid_water"],
            removeCMMotion=config["remove_cm_motion"],
            hydrogenMass=config["hydrogen_mass_amu"] * unit.amu
            if config["hydrogen_mass_amu"] is not None
            else None,
        )
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(config["use_dispersion_correction"])
                apply_tip3p_nonbonded_settings(
                    force,
                    use_switching_function=config["use_switching_function"],
                    switching_distance_nm=config["switching_distance_nm"],
                    cutoff_nm=config["nonbonded_cutoff_nm"],
                )

    if use_barostat:
        barostat = mm.MonteCarloBarostat(
            config["pressure_atm"] * unit.atmosphere,
            config["temperature_K"] * unit.kelvin,
            config["barostat_interval_steps"],
        )
        system.addForce(barostat)

    return system


def create_integrator(config: dict) -> mm.LangevinMiddleIntegrator:
    """Langevin thermostat integrator with configurable timestep and friction."""
    return mm.LangevinMiddleIntegrator(
        config["temperature_K"] * unit.kelvin,
        config["friction_per_ps"] / unit.picosecond,
        config["timestep_ps"] * unit.picoseconds,
    )


def create_simulation(
    topology: app.Topology,
    system: mm.System,
    config: dict,
) -> app.Simulation:
    """Create a :class:`openmm.app.Simulation` on the requested platform."""
    integrator = create_integrator(config)
    platform = mm.Platform.getPlatformByName(config["platform"])
    properties = dict(config.get("platform_properties", {}))
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    return simulation


def initialize_simulation_state(
    simulation: app.Simulation,
    topology: app.Topology,
    positions: unit.Quantity,
) -> None:
    """Set positions and periodic box vectors on a new simulation context."""
    simulation.context.setPositions(positions)
    simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())


def attach_reporters(
    simulation: app.Simulation,
    config: dict,
    *,
    dcd_path: Path,
    log_path: Path,
    dcd_interval_ps: float,
    log_interval_ps: float,
) -> None:
    """Attach DCD trajectory and state-data reporters for one MD phase."""
    timestep_ps = config["timestep_ps"]
    dcd_interval = ps_to_steps(dcd_interval_ps, timestep_ps)
    log_interval = ps_to_steps(log_interval_ps, timestep_ps)

    simulation.reporters.append(
        app.DCDReporter(str(dcd_path), dcd_interval)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            str(log_path),
            log_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            speed=True,
        )
    )


def clear_reporters(simulation: app.Simulation) -> None:
    """Remove all reporters (needed when switching MD phases)."""
    simulation.reporters.clear()


def run_phase(
    simulation: app.Simulation,
    n_steps: int,
    phase_name: str,
) -> float:
    """Integrate ``n_steps`` and return wall-clock time in seconds."""
    print(f"Running {phase_name}: {n_steps} steps ({n_steps * simulation.integrator.getStepSize().value_in_unit(unit.picoseconds):.1f} ps)")
    t0 = time.perf_counter()
    simulation.step(n_steps)
    elapsed = time.perf_counter() - t0
    print(f"  finished in {elapsed:.1f} s")
    return elapsed


def copy_state_to_simulation(source: app.Simulation, target: app.Simulation) -> None:
    """Copy positions, velocities, and box vectors from one simulation to another.

    ``enforcePeriodicBox`` is intentionally ``False``: the TorchFF system contains
    only a ``TorchForce`` (no bonds), so wrapping would split water molecules across
    the periodic boundary and make the intramolecular ``HarmonicBond`` term diverge.
    """
    state = source.context.getState(
        getPositions=True,
        getVelocities=True,
        enforcePeriodicBox=False,
    )
    target.context.setState(state)


def run_md_protocol(config: dict) -> None:
    """Execute the full minimization / NVT / NPT / production workflow."""
    pdb_path = Path(config["pdb_path"]).resolve()
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = "TorchFF (OpenMM-Torch)" if config["use_torchff"] else "native OpenMM"
    print(f"Force field backend: {backend}")
    if config["use_torchff"]:
        print(f"  TorchFF vdw_taper: {config['vdw_taper']}")
    print(f"LJ dispersion correction: {config['use_dispersion_correction']}")
    print(
        "LJ switching / taper: "
        f"{config['use_switching_function']}"
        + (
            f" (switch at {config['switching_distance_nm']:.2f} nm)"
            if config["use_switching_function"]
            else ""
        )
    )
    print(f"Nonbonded cutoff: {config['nonbonded_cutoff_nm']:.2f} nm")
    print(f"Rigid water: {config['rigid_water']}")
    print(f"Input PDB:  {pdb_path}")
    print(f"Output dir: {output_dir}")

    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    positions = pdb.positions

    if config.get("validate_energy", True):
        validate_torchff_vs_openmm(config, pdb_path)

    torchff_bundle = (
        prepare_torchff_bundle(config, pdb_path) if config["use_torchff"] else None
    )

    timestep_ps = config["timestep_ps"]
    nvt_steps = ps_to_steps(config["nvt_duration_ps"], timestep_ps)
    npt_steps = ps_to_steps(config["npt_duration_ps"], timestep_ps)
    prod_steps = ps_to_steps(config["production_duration_ps"], timestep_ps)

    # --- minimization + NVT (no barostat) ---
    nvt_system = build_system(
        topology, config, use_barostat=False, torchff_bundle=torchff_bundle
    )
    nvt_sim = create_simulation(topology, nvt_system, config)
    initialize_simulation_state(nvt_sim, topology, positions)

    print("Energy minimization")
    nvt_sim.minimizeEnergy(
        tolerance=config["minimize_tolerance_kj_mol_nm"] * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=config["minimize_max_iterations"],
    )

    print(f"Setting velocities to {config['temperature_K']} K")
    nvt_sim.context.setVelocitiesToTemperature(config["temperature_K"] * unit.kelvin)

    attach_reporters(
        nvt_sim,
        config,
        dcd_path=output_dir / config["nvt_dcd_filename"],
        log_path=output_dir / config["nvt_log_filename"],
        dcd_interval_ps=config["nvt_npt_dcd_interval_ps"],
        log_interval_ps=config["nvt_npt_log_interval_ps"],
    )
    run_phase(nvt_sim, nvt_steps, "NVT equilibration")
    clear_reporters(nvt_sim)

    # --- NPT equilibration ---
    npt_system = build_system(
        topology, config, use_barostat=True, torchff_bundle=torchff_bundle
    )
    npt_sim = create_simulation(topology, npt_system, config)
    copy_state_to_simulation(nvt_sim, npt_sim)

    attach_reporters(
        npt_sim,
        config,
        dcd_path=output_dir / config["npt_dcd_filename"],
        log_path=output_dir / config["npt_log_filename"],
        dcd_interval_ps=config["nvt_npt_dcd_interval_ps"],
        log_interval_ps=config["nvt_npt_log_interval_ps"],
    )
    run_phase(npt_sim, npt_steps, "NPT equilibration")
    clear_reporters(npt_sim)

    # --- NPT production (continue from last NPT snapshot) ---
    attach_reporters(
        npt_sim,
        config,
        dcd_path=output_dir / config["production_dcd_filename"],
        log_path=output_dir / config["production_log_filename"],
        dcd_interval_ps=config["production_dcd_interval_ps"],
        log_interval_ps=config["production_log_interval_ps"],
    )
    run_phase(npt_sim, prod_steps, "NPT production")

    print("MD protocol complete.")
    print(f"Trajectories and logs written to {output_dir}")


# ---------------------------------------------------------------------------
# All simulation settings (edit here or override before calling run_md_protocol)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

CONFIG: dict = {
    # Backend toggle
    "use_torchff": False,
    "use_customized_ops": True,
    "use_cuda_graphs": False,
    "cuda_graph_warmup_steps": 10,
    # Input / output
    "pdb_path": _REPO_ROOT / "water_300.pdb",
    "output_dir_openmm": _SCRIPT_DIR / "md_openmm",
    "output_dir_torchff": _SCRIPT_DIR / "md_torchff",
    "nvt_dcd_filename": "nvt.dcd",
    "nvt_log_filename": "nvt.log",
    "npt_dcd_filename": "npt.dcd",
    "npt_log_filename": "npt.log",
    "production_dcd_filename": "production.dcd",
    "production_log_filename": "production.log",
    # Force field and nonbonded
    "forcefield": "tip3p.xml",
    "nonbonded_method": "PME",
    "nonbonded_cutoff_nm": 0.8,
    "use_dispersion_correction": False,
    "vdw_taper": False,
    "use_switching_function": False,
    "switching_distance_nm": 0.7,
    "validate_energy": True,
    "constraints": None,
    "rigid_water": False,
    "hydrogen_mass_amu": None,  # no HMR
    "remove_cm_motion": False,
    # Integrator
    "timestep_ps": 0.001,  # 1 fs
    "temperature_K": 298.15,
    "friction_per_ps": 1.0,
    # Barostat (NPT phases)
    "pressure_atm": 1.0,
    "barostat_interval_steps": 25,
    # Phase lengths
    "nvt_duration_ps": 500.0,
    "npt_duration_ps": 500.0,
    "production_duration_ps": 5000.0,  # 5 ns
    # Trajectory and log intervals
    "nvt_npt_dcd_interval_ps": 10.0,
    "nvt_npt_log_interval_ps": 10.0,
    "production_dcd_interval_ps": 5.0,
    "production_log_interval_ps": 5.0,
    # Minimization
    "minimize_tolerance_kj_mol_nm": 10.0,
    "minimize_max_iterations": 0,  # 0 = OpenMM default
    # Platform
    "platform": "CUDA",
    "platform_properties": {"CudaPrecision": "mixed"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TIP3P production MD with OpenMM or TorchFF.")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--torchff",
        action="store_true",
        help="Use TorchFF via OpenMM-Torch (output: md_torchff/).",
    )
    backend.add_argument(
        "--openmm",
        action="store_true",
        help="Use native OpenMM TIP3P (output: md_openmm/).",
    )
    parser.add_argument(
        "--pdb-path",
        type=Path,
        default=None,
        help="Input water box PDB (default: examples/water_300.pdb).",
    )
    parser.add_argument(
        "--output-dir-openmm",
        type=Path,
        default=None,
        help="Override native OpenMM output directory (default: md_openmm/).",
    )
    parser.add_argument(
        "--dispersion-correction",
        action="store_true",
        help="Enable OpenMM LJ tail / long-range dispersion correction on NonbondedForce.",
    )
    parser.add_argument(
        "--forcebalance-protocol",
        action="store_true",
        help=(
            "Use ForceBalance-like OpenMM nonbonded settings: 0.85 nm cutoff, "
            "LJ switching at 0.75 nm, dispersion correction; output to md_openmm_fb/."
        ),
    )
    parser.add_argument(
        "--rigid-water",
        action="store_true",
        help="Use OpenMM rigidWater=True (SHAKE on water geometry).",
    )
    taper = parser.add_mutually_exclusive_group()
    taper.add_argument(
        "--vdw-taper",
        dest="vdw_taper",
        action="store_true",
        default=None,
        help="Enable OpenMM LJ switching and TorchFF vdW taper.",
    )
    taper.add_argument(
        "--no-vdw-taper",
        dest="vdw_taper",
        action="store_false",
        help="Hard vdW cutoff (no taper).",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip TorchFF vs OpenMM energy validation before MD.",
    )
    return parser.parse_args()


def apply_vdw_taper_settings(config: dict) -> dict:
    """Enable OpenMM switching and TorchFF vdW taper with default switch distance."""
    config = dict(config)
    config["vdw_taper"] = True
    config["use_switching_function"] = True
    if config["switching_distance_nm"] is None:
        config["switching_distance_nm"] = config["nonbonded_cutoff_nm"] - 0.1
    return config


def apply_forcebalance_protocol(config: dict) -> dict:
    """Apply ForceBalance OpenMM liquid nonbonded defaults."""
    config = dict(config)
    config["nonbonded_cutoff_nm"] = 0.85
    config["use_dispersion_correction"] = True
    config["use_switching_function"] = True
    config["switching_distance_nm"] = config["nonbonded_cutoff_nm"] - 0.1
    if not config["use_torchff"]:
        config["output_dir_openmm"] = _SCRIPT_DIR / "md_openmm_fb"
    return config


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)
    if args.torchff:
        config["use_torchff"] = True
    elif args.openmm:
        config["use_torchff"] = False
    if args.pdb_path is not None:
        config["pdb_path"] = args.pdb_path
    if args.dispersion_correction:
        config["use_dispersion_correction"] = True
    if args.forcebalance_protocol:
        config = apply_forcebalance_protocol(config)
    if args.output_dir_openmm is not None:
        config["output_dir_openmm"] = args.output_dir_openmm
    if args.rigid_water:
        config["rigid_water"] = True
    if args.vdw_taper is not None:
        config["vdw_taper"] = args.vdw_taper
        config["use_switching_function"] = args.vdw_taper
    if config["vdw_taper"]:
        config = apply_vdw_taper_settings(config)
        if config["use_torchff"]:
            config["output_dir_torchff"] = _SCRIPT_DIR / "md_torchff_taper"
        else:
            config["output_dir_openmm"] = _SCRIPT_DIR / "md_openmm_taper"
    if args.skip_validate:
        config["validate_energy"] = False
    run_md_protocol(config)


if __name__ == "__main__":
    main()
