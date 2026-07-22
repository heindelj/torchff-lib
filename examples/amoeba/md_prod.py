"""
OpenMM production MD for a periodic AMOEBA2018 water box.

Loads ``examples/water_300.pdb`` and runs:

1. Energy minimization
2. NVT equilibration at 298.15 K (1 fs timestep, 500 ps)
3. NPT equilibration with a Monte Carlo barostat (1 fs, 500 ps)
4. NPT production (1 fs, 5 ns)

Trajectories and logs are written to ``examples/amoeba/md_openmm/`` (native OpenMM)
or ``examples/amoeba/md_torchff/`` when :data:`CONFIG` ``use_torchff`` is enabled.

All simulation parameters live in :data:`CONFIG` at the bottom of this file.

Run on a NERSC GPU compute node (interactive or batch)::

    srun --nodes 1 --qos interactive --time 2:00:00 --constraint gpu --gres=gpu:1 \\
        bash -c 'module load conda && mamba activate openmm-torch-py312-cu124 && \\
        python examples/amoeba/md_prod.py --torchff'

Or from ``examples/amoeba``::

    python md_prod.py
    python md_prod.py --torchff
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import torch

from md import (
    build_amoeba_torchff_config,
    create_amoeba_torchforce,
    openmm_system_with_amoeba_torchforce,
)
from model import TorchFFAmoeba

if TYPE_CHECKING:
    from openmmtorch import TorchForce


@dataclass(frozen=True)
class TorchFFBundle:
    """Cached TorchFF model shared across MD phases."""

    model: TorchFFAmoeba


def create_torchforce(config: dict, model: TorchFFAmoeba) -> TorchForce:
    """Build a fresh OpenMM-Torch force (one per OpenMM System)."""
    return create_amoeba_torchforce(
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
    """Build :class:`TorchFFAmoeba` on CUDA from the input PDB and cutoff."""
    dtype = config["dtype"]
    torch.set_default_dtype(dtype)
    cfg, _, _ = build_amoeba_torchff_config(pdb_path, config["nonbonded_cutoff_nm"])
    model = TorchFFAmoeba(
        cfg,
        use_customized_ops=config["use_customized_ops"],
        vdw_taper=config["vdw_taper"],
        pme_customized=config["pme_customized"],
    )
    device = torch.device("cuda")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return TorchFFBundle(model=model)


def build_system(
    topology: app.Topology,
    config: dict,
    *,
    use_barostat: bool,
    torchff_bundle: TorchFFBundle | None = None,
) -> mm.System:
    """Build an AMOEBA system (native OpenMM or TorchFF) with optional Monte Carlo barostat."""
    if config["use_torchff"]:
        if torchff_bundle is None:
            raise ValueError("torchff_bundle is required when use_torchff=True")
        torch_force = create_torchforce(config, torchff_bundle.model)
        system = openmm_system_with_amoeba_torchforce(
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
            polarization=config["polarization"],
        )
        for force in system.getForces():
            if isinstance(force, mm.AmoebaVdwForce):
                force.setUseDispersionCorrection(config["use_dispersion_correction"])

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
    if config.get("use_torchff") and config.get("dtype") is torch.float64:
        properties["CudaPrecision"] = "double"
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


def ns_per_day(n_steps: int, timestep_ps: float, elapsed_s: float) -> float:
    """Simulated nanoseconds per day from wall time and integration steps."""
    if elapsed_s <= 0.0:
        return 0.0
    sim_ns = n_steps * timestep_ps * 1e-3
    return sim_ns / elapsed_s * 86400.0


def run_phase(
    simulation: app.Simulation,
    n_steps: int,
    phase_name: str,
    *,
    timestep_ps: float,
) -> float:
    """Integrate ``n_steps`` and return wall-clock time in seconds."""
    duration_ps = n_steps * timestep_ps
    print(f"Running {phase_name}: {n_steps} steps ({duration_ps:.1f} ps)")
    t0 = time.perf_counter()
    simulation.step(n_steps)
    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / n_steps * 1e3 if n_steps else 0.0
    print(
        f"  finished in {elapsed:.1f} s "
        f"({ms_per_step:.3f} ms/step, {ns_per_day(n_steps, timestep_ps, elapsed):.2f} ns/day)"
    )
    return elapsed


def copy_state_to_simulation(source: app.Simulation, target: app.Simulation) -> None:
    """Copy positions, velocities, and box vectors from one simulation to another.

    ``enforcePeriodicBox`` is intentionally ``False``: the TorchFF system contains
    only a ``TorchForce`` (no valence forces), so wrapping would split water molecules
    across the periodic boundary.
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
        print(
            f"  TorchFF: dtype={config['dtype']}, vdw_taper={config['vdw_taper']}, "
            f"customized_ops={config['use_customized_ops']}, "
            f"pme_customized={config['pme_customized']}"
        )
    print(f"Input PDB:  {pdb_path}")
    print(f"Output dir: {output_dir}")

    pdb = app.PDBFile(str(pdb_path))
    topology = pdb.topology
    positions = pdb.positions

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
    run_phase(nvt_sim, nvt_steps, "NVT equilibration", timestep_ps=timestep_ps)
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
    run_phase(npt_sim, npt_steps, "NPT equilibration", timestep_ps=timestep_ps)
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
    run_phase(npt_sim, prod_steps, "NPT production", timestep_ps=timestep_ps)

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
    "pme_customized": True,
    "dtype": torch.float64,
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
    # Force field and nonbonded (matches examples/amoeba/md.py)
    "forcefield": "amoeba2018.xml",
    "nonbonded_method": "PME",
    "nonbonded_cutoff_nm": 0.8,
    "polarization": "direct",
    "use_dispersion_correction": False,
    "vdw_taper": False,  # hard vdW cutoff (TorchFF); mirrors tip3p setUseSwitchingFunction(False)
    "constraints": None,
    "rigid_water": False,
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
    parser = argparse.ArgumentParser(description="AMOEBA production MD with OpenMM or TorchFF.")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--torchff",
        action="store_true",
        help="Use TorchFF via OpenMM-Torch (output: md_torchff/).",
    )
    backend.add_argument(
        "--openmm",
        action="store_true",
        help="Use native OpenMM AMOEBA (output: md_openmm/).",
    )
    taper = parser.add_mutually_exclusive_group()
    taper.add_argument(
        "--vdw-taper",
        dest="vdw_taper",
        action="store_true",
        default=None,
        help="Apply OpenMM vdW taper in TorchFF (default for --torchff).",
    )
    taper.add_argument(
        "--no-vdw-taper",
        dest="vdw_taper",
        action="store_false",
        help="Use a hard vdW cutoff in TorchFF (no taper).",
    )
    parser.add_argument(
        "--cutoff-nm",
        type=float,
        default=None,
        help="Nonbonded cutoff (nm) for TorchFF config build.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="Torch dtype for TorchFF model and TorchForce trace.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Short smoke test: tiny EM/NVT/NPT/production and frequent reporters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)
    if args.torchff:
        config["use_torchff"] = True
    elif args.openmm:
        config["use_torchff"] = False
    if args.vdw_taper is not None:
        config["vdw_taper"] = args.vdw_taper
    if args.cutoff_nm is not None:
        config["nonbonded_cutoff_nm"] = args.cutoff_nm
    if args.dtype is not None:
        config["dtype"] = torch.float32 if args.dtype == "float32" else torch.float64
    if config["use_torchff"]:
        config["output_dir_torchff"] = _SCRIPT_DIR / (
            "md_torchff_taper" if config["vdw_taper"] else "md_torchff_notaper"
        )
    if args.test:
        config.update(
            nvt_duration_ps=2.0,
            npt_duration_ps=2.0,
            production_duration_ps=2.0,
            nvt_npt_dcd_interval_ps=1.0,
            nvt_npt_log_interval_ps=1.0,
            production_dcd_interval_ps=1.0,
            production_log_interval_ps=1.0,
            minimize_max_iterations=50,
        )
        if config["use_torchff"]:
            suffix = "_test"
            config["output_dir_torchff"] = Path(config["output_dir_torchff"]).parent / (
                Path(config["output_dir_torchff"]).name + suffix
            )
        else:
            suffix = "_test"
            config["output_dir_openmm"] = Path(config["output_dir_openmm"]).parent / (
                Path(config["output_dir_openmm"]).name + suffix
            )
    run_md_protocol(config)


if __name__ == "__main__":
    main()
