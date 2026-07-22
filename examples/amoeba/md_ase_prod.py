"""
ASE production MD for a periodic AMOEBA2018 water box, driven by the TorchFF AMOEBA model.

This mirrors the OpenMM workflow in ``examples/amoeba/md_prod.py`` (EM -> NVT -> NPT ->
production) but runs entirely through ASE (:mod:`ase.md`) on top of
:class:`md_ase.AmoebaTorchFFCalculator`, which wraps :class:`model.TorchFFAmoeba` in **full
customized CUDA ops**. The pressure coupling is a Monte Carlo barostat implemented here
(:class:`MonteCarloBarostatLangevin`), inspired by ``pycmm-dev/cmm/ase`` but rewritten so it
scales molecular centers of mass (matching OpenMM ``MonteCarloBarostat`` rather than scaling all
atoms, which would strain intramolecular bonds).

Phases
------
1. Energy minimization (:class:`ase.optimize.LBFGS`, fixed cell).
2. NVT equilibration (:class:`ase.md.langevin.Langevin`).
3. NPT equilibration (Langevin + Monte Carlo barostat).
4. Production (NPT by default; NVT optional).

Each MD phase writes an ASE ``.traj`` trajectory and a text log (step, time, temperature,
potential/total energy, density, wall clock). All parameters live in :data:`CONFIG` and can be
overridden on the command line.

Run on a NERSC GPU node with the ``openmm-torch-py312-cu124`` env. Examples::

    python md_ase_prod.py --test                 # short smoke test (taper)
    python md_ase_prod.py --vdw-taper            # full 5 ns run, tapered vdW
    python md_ase_prod.py --no-vdw-taper         # full 5 ns run, hard-cutoff vdW
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import LBFGS
from ase.units import bar, fs, kB

_AMOEBA_DIR = Path(__file__).resolve().parent
if str(_AMOEBA_DIR) not in sys.path:
    sys.path.insert(0, str(_AMOEBA_DIR))

from md import build_amoeba_torchff_config, default_water_pdb_path
from md_ase import AmoebaTorchFFCalculator
from model import TorchFFAmoeba

# 1 amu in grams and 1 Angstrom^3 in cm^3, for density in g/cm^3.
_AMU_TO_GRAM = 1.66053906660e-24
_ANG3_TO_CM3 = 1.0e-24


class MonteCarloBarostatLangevin(Langevin):
    """Langevin dynamics with an isotropic Monte Carlo barostat (NPT).

    A volume move is attempted every ``barostat_interval`` steps: the box and the molecular
    centers of mass are scaled by ``(V'/V)^(1/3)`` while internal geometry is preserved, then the
    move is accepted with the OpenMM ``MonteCarloBarostat`` criterion

    .. math::

        w = \\Delta E + P\\,\\Delta V - N_\\mathrm{mol}\\,k_BT\\,\\ln\\!\\frac{V'}{V},

    accepting when ``w <= 0`` or ``rand() < exp(-w / k_BT)``. The step size ``volume_scale`` is
    adapted toward a 25-75% acceptance window, as in OpenMM.

    Inspired by ``pycmm-dev/cmm/ase/LangevinMonteCarloBarostat.py`` (rewritten for molecular,
    rather than per-atom, scaling).
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        *,
        temperature_K: float,
        friction: float,
        pressure_bar: float = 1.01325,
        barostat_interval: int = 25,
        atoms_per_molecule: int = 3,
        fixcm: bool = True,
        rng=None,
        **kwargs,
    ) -> None:
        super().__init__(
            atoms,
            timestep,
            temperature_K=temperature_K,
            friction=friction,
            fixcm=fixcm,
            rng=rng,
            **kwargs,
        )
        self.pressure = pressure_bar * bar  # ASE pressure units (eV / Angstrom^3)
        self.barostat_interval = int(barostat_interval)
        self.atoms_per_molecule = int(atoms_per_molecule)
        self.volume_scale = atoms.get_volume() * 0.01
        self.num_attempted = 0
        self.num_accepted = 0

    def _molecular_com_positions(self, positions: np.ndarray, masses: np.ndarray):
        m = self.atoms_per_molecule
        n_mol = len(self.atoms) // m
        pos_r = positions.reshape(n_mol, m, 3)
        mass_r = masses.reshape(n_mol, m)
        com = (pos_r * mass_r[:, :, None]).sum(axis=1) / mass_r.sum(axis=1)[:, None]
        return pos_r, com, n_mol

    def _apply_length_scale(self, length_scale: float) -> None:
        """Scale the cell and molecular COMs by ``length_scale`` (keeps internal geometry)."""
        positions = self.atoms.get_positions()
        masses = self.atoms.get_masses()
        pos_r, com, _ = self._molecular_com_positions(positions, masses)
        shift = (com * length_scale - com)[:, None, :]
        new_positions = (pos_r + shift).reshape(len(self.atoms), 3)
        self.atoms.set_cell(self.atoms.get_cell() * length_scale, scale_atoms=False)
        self.atoms.set_positions(new_positions)

    def step(self, forces=None):
        forces = super().step(forces)
        if self.get_number_of_steps() == 0:
            return forces
        if self.get_number_of_steps() % self.barostat_interval != 0:
            return forces

        m = self.atoms_per_molecule
        n_mol = len(self.atoms) // m

        old_cell = self.atoms.get_cell().copy()
        old_positions = self.atoms.get_positions().copy()
        old_volume = self.atoms.get_volume()
        old_energy = self.atoms.get_potential_energy()

        d_volume = self.volume_scale * self.rng.uniform(-1.0, 1.0)
        new_volume = old_volume + d_volume
        if new_volume <= 0.0:
            return forces
        length_scale = (new_volume / old_volume) ** (1.0 / 3.0)

        self._apply_length_scale(length_scale)
        new_energy = self.atoms.get_potential_energy()

        kT = self.temp  # ASE Langevin stores temperature as kB * T_K (eV)
        d_energy = new_energy - old_energy
        work = d_energy + self.pressure * d_volume - n_mol * kT * np.log(new_volume / old_volume)

        accept = (work <= 0.0) or (self.rng.uniform() < np.exp(-work / kT))
        if accept:
            self.num_accepted += 1
        else:
            self.atoms.set_cell(old_cell, scale_atoms=False)
            self.atoms.set_positions(old_positions)
        self.num_attempted += 1

        if self.num_attempted >= 10:
            if self.num_accepted < 0.25 * self.num_attempted:
                self.volume_scale /= 1.1
                self.num_attempted = 0
                self.num_accepted = 0
            elif self.num_accepted > 0.75 * self.num_attempted:
                self.volume_scale = min(self.volume_scale * 1.1, old_volume * 0.3)
                self.num_attempted = 0
                self.num_accepted = 0

        return forces


class MDPhaseLogger:
    """Append per-interval thermodynamic data for one MD phase to a text log (and optionally stdout)."""

    _HEADER = (
        "# step      time(ps)   temperature(K)   pot_energy(eV)   tot_energy(eV)   "
        "density(g/cm^3)   volume(A^3)   wallclock"
    )

    def __init__(self, filename: os.PathLike, atoms: Atoms, dynamics, verbose: bool = True,
                 append: bool = False) -> None:
        self.atoms = atoms
        self.dynamics = dynamics
        self.verbose = verbose
        self._file = open(filename, "a" if append else "w")
        if not append:
            self._file.write(self._HEADER + "\n")
            self._file.flush()
        if verbose:
            print(self._HEADER)
        self._start = time.time()
        self._total_mass_amu = float(np.sum(atoms.get_masses()))

    def __call__(self) -> None:
        step = self.dynamics.get_number_of_steps()
        time_ps = self.dynamics.get_time() / (1000.0 * fs)
        temperature = self.atoms.get_temperature()
        e_pot = self.atoms.get_potential_energy()
        e_kin = self.atoms.get_kinetic_energy()
        e_tot = e_pot + e_kin
        volume = self.atoms.get_volume()
        density = self._total_mass_amu * _AMU_TO_GRAM / (volume * _ANG3_TO_CM3)

        elapsed = int(time.time() - self._start)
        hrs, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        wall = f"{hrs:02d}:{mins:02d}:{secs:02d}"

        msg = (
            f"{step:8d}  {time_ps:10.3f}  {temperature:13.2f}  {e_pot:15.6f}  "
            f"{e_tot:15.6f}  {density:14.6f}  {volume:12.3f}  {wall}"
        )
        self._file.write(msg + "\n")
        self._file.flush()
        if self.verbose:
            print(msg)

    def close(self) -> None:
        self._file.close()


def build_amoeba_calculator(config: dict, pdb_path: Path):
    """Build the TorchFF AMOEBA ASE calculator and the initial periodic box (nm) from a PDB."""
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    dtype = config["dtype"]

    cfg, _topology, _ref = build_amoeba_torchff_config(pdb_path, config["nonbonded_cutoff_nm"])
    model = TorchFFAmoeba(
        cfg,
        use_customized_ops=config["use_customized_ops"],
        vdw_taper=config["vdw_taper"],
        pme_customized=config["pme_customized"],
    ).to(device, dtype=dtype)
    model.eval()

    calc = AmoebaTorchFFCalculator(model, device=device, dtype=dtype)
    cell_ang = cfg.initial_box_nm.detach().cpu().to(torch.float64).numpy() * 10.0
    return calc, cell_ang


def prepare_atoms(pdb_path: Path, cell_ang: np.ndarray, calc) -> Atoms:
    """Read the PDB into an ASE :class:`~ase.Atoms`, set the periodic cell, and attach the calculator."""
    atoms = read(str(pdb_path))
    atoms.pbc = True
    atoms.set_cell(cell_ang)
    atoms.calc = calc
    return atoms


def friction_in_ase_units(friction_per_ps: float) -> float:
    """Convert a friction coefficient in 1/ps to ASE's 1/(ASE time unit) convention."""
    return (friction_per_ps / 1000.0) / fs


def run_minimization(atoms: Atoms, config: dict, workdir: Path) -> None:
    if config["em_steps"] <= 0:
        return
    print(f"Energy minimization: <= {config['em_steps']} steps, fmax={config['em_fmax']} eV/A")
    atoms.calc.set_minimization() if hasattr(atoms.calc, "set_minimization") else None
    opt = LBFGS(atoms, logfile=str(workdir / "em.log"))
    opt.run(fmax=config["em_fmax"], steps=config["em_steps"])
    atoms.write(str(workdir / "em.xyz"))
    print("  minimization finished")


_CHECKPOINT_NAME = "checkpoint.npz"


def save_checkpoint(workdir: Path, phase: str, nsteps_done: int, atoms: Atoms,
                    volume_scale: float | None = None) -> None:
    """Atomically write a restart checkpoint (phase, step, positions, velocities, cell)."""
    tmp = workdir / "checkpoint.tmp.npz"
    np.savez(
        tmp,
        phase=np.array(phase),
        nsteps_done=np.array(int(nsteps_done)),
        positions=atoms.get_positions(),
        velocities=atoms.get_velocities(),
        cell=atoms.get_cell().array,
        volume_scale=np.array(-1.0 if volume_scale is None else float(volume_scale)),
    )
    os.replace(tmp, workdir / _CHECKPOINT_NAME)


def load_checkpoint(workdir: Path) -> dict | None:
    """Load a restart checkpoint if present."""
    path = workdir / _CHECKPOINT_NAME
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "phase": str(data["phase"]),
        "nsteps_done": int(data["nsteps_done"]),
        "positions": data["positions"],
        "velocities": data["velocities"],
        "cell": data["cell"],
        "volume_scale": float(data["volume_scale"]),
    }


class CheckpointSaver:
    """Periodically write a restart checkpoint for one MD phase."""

    def __init__(self, workdir: Path, phase: str, dynamics, atoms: Atoms) -> None:
        self.workdir = workdir
        self.phase = phase
        self.dynamics = dynamics
        self.atoms = atoms

    def __call__(self) -> None:
        vs = getattr(self.dynamics, "volume_scale", None)
        save_checkpoint(self.workdir, self.phase, self.dynamics.get_number_of_steps(), self.atoms, vs)


def run_md_phase(
    atoms: Atoms,
    config: dict,
    workdir: Path,
    *,
    phase: str,
    n_steps: int,
    use_barostat: bool,
    traj_filename: str,
    log_filename: str,
    traj_interval: int,
    log_interval: int,
    steps_already: int = 0,
    volume_scale: float | None = None,
) -> float:
    """Run one Langevin (NVT) or Langevin+MC-barostat (NPT) phase; return wall-clock seconds.

    ``steps_already`` resumes a partially completed phase (trajectory/log are appended and the step
    counter continues), enabling checkpoint/restart across time-limited Slurm jobs.
    """
    remaining = n_steps - steps_already
    if remaining <= 0:
        return 0.0
    resuming = steps_already > 0
    timestep = config["timestep_fs"] * fs
    friction = friction_in_ase_units(config["friction_per_ps"])

    if use_barostat:
        dyn = MonteCarloBarostatLangevin(
            atoms,
            timestep,
            temperature_K=config["temperature_K"],
            friction=friction,
            pressure_bar=config["pressure_bar"],
            barostat_interval=config["barostat_interval_steps"],
            atoms_per_molecule=config["atoms_per_molecule"],
        )
        if volume_scale is not None and volume_scale > 0:
            dyn.volume_scale = volume_scale
    else:
        dyn = Langevin(
            atoms,
            timestep,
            temperature_K=config["temperature_K"],
            friction=friction,
        )
    dyn.nsteps = steps_already

    traj = Trajectory(str(workdir / traj_filename), "a" if resuming else "w", atoms)
    dyn.attach(traj.write, interval=traj_interval)
    logger = MDPhaseLogger(workdir / log_filename, atoms, dyn, verbose=config["verbose"],
                           append=resuming)
    dyn.attach(logger, interval=log_interval)
    dyn.attach(CheckpointSaver(workdir, phase, dyn, atoms),
               interval=config["checkpoint_interval"])

    duration_ps = remaining * config["timestep_fs"] * 1e-3
    print(f"Running {phase}: {remaining} steps (resume from {steps_already}/{n_steps}, "
          f"{duration_ps:.1f} ps), barostat={use_barostat}")
    t0 = time.perf_counter()
    dyn.run(remaining)
    elapsed = time.perf_counter() - t0

    logger.close()
    traj.close()
    save_checkpoint(workdir, phase, n_steps, atoms, getattr(dyn, "volume_scale", None))
    atoms.write(str(workdir / f"{phase.lower().replace(' ', '_')}.xyz"))
    ns_day = (duration_ps * 1e-3) / elapsed * 86400.0 if elapsed > 0 else 0.0
    print(
        f"  {phase} finished in {elapsed:.1f} s "
        f"({elapsed / remaining * 1e3:.3f} ms/step, {ns_day:.2f} ns/day)"
    )
    return elapsed


def run_workflow(config: dict) -> None:
    pdb_path = Path(config["pdb_path"]).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Missing PDB: {pdb_path}")
    workdir = Path(config["output_dir"]).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"TorchFF AMOEBA (ASE) production workflow")
    print(f"  PDB:            {pdb_path}")
    print(f"  Output dir:     {workdir}")
    print(f"  vdw_taper:      {config['vdw_taper']}")
    print(f"  customized ops: {config['use_customized_ops']} (PME customized={config['pme_customized']})")
    print(f"  dtype/device:   {config['dtype']} / {config['device']}")

    calc, cell_ang = build_amoeba_calculator(config, pdb_path)
    atoms = prepare_atoms(pdb_path, cell_ang, calc)

    # MD phases in order; checkpoint/restart resumes the first incomplete phase.
    md_phases = [
        dict(phase="NVT", n_steps=config["nvt_steps"], use_barostat=False,
             traj_filename="nvt.traj", log_filename="nvt.log",
             traj_interval=config["equil_traj_interval"], log_interval=config["equil_log_interval"]),
        dict(phase="NPT", n_steps=config["npt_steps"], use_barostat=True,
             traj_filename="npt.traj", log_filename="npt.log",
             traj_interval=config["equil_traj_interval"], log_interval=config["equil_log_interval"]),
        dict(phase="Production", n_steps=config["prod_steps"],
             use_barostat=not config["prod_nvt"],
             traj_filename="production.traj", log_filename="production.log",
             traj_interval=config["prod_traj_interval"], log_interval=config["prod_log_interval"]),
    ]
    phase_order = [p["phase"] for p in md_phases]

    ckpt = load_checkpoint(workdir)
    if ckpt is None:
        e0 = atoms.get_potential_energy()
        print(f"  initial potential energy: {e0:.6f} eV")
        run_minimization(atoms, config, workdir)
        if hasattr(calc, "unset_minimization"):
            calc.unset_minimization()
        MaxwellBoltzmannDistribution(atoms, temperature_K=config["temperature_K"], force_temp=True)
        Stationary(atoms)
        start_idx, start_step, start_vs = 0, 0, None
    else:
        print(f"  resuming from checkpoint: phase={ckpt['phase']} step={ckpt['nsteps_done']}")
        if hasattr(calc, "unset_minimization"):
            calc.unset_minimization()
        atoms.set_cell(ckpt["cell"])
        atoms.set_positions(ckpt["positions"])
        atoms.set_velocities(ckpt["velocities"])
        ck_idx = phase_order.index(ckpt["phase"])
        if ckpt["nsteps_done"] >= md_phases[ck_idx]["n_steps"]:
            start_idx, start_step, start_vs = ck_idx + 1, 0, None
        else:
            start_idx, start_step = ck_idx, ckpt["nsteps_done"]
            start_vs = ckpt["volume_scale"] if ckpt["volume_scale"] > 0 else None

    for idx in range(start_idx, len(md_phases)):
        spec = md_phases[idx]
        steps_already = start_step if idx == start_idx else 0
        vs = start_vs if idx == start_idx else None
        run_md_phase(atoms, config, workdir, steps_already=steps_already, volume_scale=vs, **spec)

    (workdir / "DONE").write_text("workflow complete\n")
    print("MD workflow complete.")
    print(f"Trajectories and logs in {workdir}")


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

CONFIG: dict = {
    # Model / backend
    "use_customized_ops": True,
    # Use the customized CUDA PME reciprocal op. Its rank-2 coordinate gradient was fixed
    # (see csrc/pme/pme.cu PMEAllFunction::backward); set to False to fall back to the
    # autograd Python PME if you ever need to cross-check.
    "pme_customized": True,
    "vdw_taper": True,
    "dtype": torch.float32,
    "device": "cuda",
    # Input / output
    "pdb_path": _REPO_ROOT / "water_300.pdb",
    "output_dir": _SCRIPT_DIR / "md_ase_taper",
    # Nonbonded (matches examples/amoeba/md.py defaults)
    "nonbonded_cutoff_nm": 1.0,
    # Thermostat / barostat
    "temperature_K": 298.15,
    "friction_per_ps": 1.0,
    "pressure_bar": 1.01325,
    "barostat_interval_steps": 25,
    "atoms_per_molecule": 3,
    # Integrator
    "timestep_fs": 1.0,
    # Minimization
    "em_steps": 2000,
    "em_fmax": 0.05,
    # Phase lengths (steps); 1 fs timestep -> 500 ps, 500 ps, 5 ns
    "nvt_steps": 500_000,
    "npt_steps": 500_000,
    "prod_steps": 5_000_000,
    "prod_nvt": False,
    # Reporter intervals (steps)
    "equil_traj_interval": 10_000,   # 10 ps
    "equil_log_interval": 10_000,    # 10 ps
    "prod_traj_interval": 5_000,     # 5 ps
    "prod_log_interval": 5_000,      # 5 ps
    # Restart checkpoint cadence (steps); enables resuming across time-limited Slurm jobs.
    "checkpoint_interval": 5_000,    # 5 ps
    "verbose": True,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASE EM+NVT+NPT+Production MD with TorchFF AMOEBA.")
    p.add_argument("-N", "--n-waters", type=int, default=None, help="Use examples/water_<N>.pdb.")
    p.add_argument("--pdb", type=Path, default=None, help="Explicit periodic PDB path.")
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    taper = p.add_mutually_exclusive_group()
    taper.add_argument("--vdw-taper", dest="vdw_taper", action="store_true", default=None,
                       help="Apply OpenMM vdW taper (default).")
    taper.add_argument("--no-vdw-taper", dest="vdw_taper", action="store_false",
                       help="Use a hard vdW cutoff (no taper).")
    p.add_argument("--cutoff-nm", type=float, default=None, help="Nonbonded cutoff (nm).")
    p.add_argument("--dtype", choices=["float32", "float64"], default=None, help="Torch dtype.")
    p.add_argument("--prod-nvt", action="store_true", help="Run production in NVT (no barostat).")
    p.add_argument(
        "--no-customized-ops",
        dest="use_customized_ops",
        action="store_false",
        default=None,
        help="Use the pure-Python reference path instead of customized CUDA ops.",
    )
    pme_grp = p.add_mutually_exclusive_group()
    pme_grp.add_argument(
        "--pme-customized",
        dest="pme_customized",
        action="store_true",
        default=None,
        help="Use the customized CUDA PME reciprocal op (default).",
    )
    pme_grp.add_argument(
        "--no-pme-customized",
        dest="pme_customized",
        action="store_false",
        default=None,
        help="Use the autograd Python PME reciprocal op (cross-check / fallback).",
    )
    p.add_argument("--em-steps", type=int, default=None, help="Override minimization steps.")
    p.add_argument("--nvt-steps", type=int, default=None, help="Override NVT steps.")
    p.add_argument("--npt-steps", type=int, default=None, help="Override NPT steps.")
    p.add_argument("--prod-steps", type=int, default=None, help="Override production steps.")
    p.add_argument("--friction-per-ps", type=float, default=None, help="Langevin friction (1/ps).")
    p.add_argument("--log-interval", type=int, default=None, help="Override all log intervals (steps).")
    p.add_argument("--traj-interval", type=int, default=None, help="Override all trajectory intervals (steps).")
    p.add_argument("--checkpoint-interval", type=int, default=None, help="Restart checkpoint cadence (steps).")
    p.add_argument(
        "--test",
        action="store_true",
        help="Short smoke test: tiny EM/NVT/NPT/production and frequent reporters.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(CONFIG)

    if args.pdb is not None:
        config["pdb_path"] = args.pdb
    elif args.n_waters is not None:
        config["pdb_path"] = default_water_pdb_path(args.n_waters)

    if args.vdw_taper is not None:
        config["vdw_taper"] = args.vdw_taper
    if args.cutoff_nm is not None:
        config["nonbonded_cutoff_nm"] = args.cutoff_nm
    if args.dtype is not None:
        config["dtype"] = torch.float32 if args.dtype == "float32" else torch.float64
    if args.prod_nvt:
        config["prod_nvt"] = True
    if args.use_customized_ops is not None:
        config["use_customized_ops"] = args.use_customized_ops
        if not args.use_customized_ops:
            config["pme_customized"] = False
    if args.pme_customized is not None:
        config["pme_customized"] = args.pme_customized
    if args.em_steps is not None:
        config["em_steps"] = args.em_steps
    if args.nvt_steps is not None:
        config["nvt_steps"] = args.nvt_steps
    if args.npt_steps is not None:
        config["npt_steps"] = args.npt_steps
    if args.prod_steps is not None:
        config["prod_steps"] = args.prod_steps
    if args.friction_per_ps is not None:
        config["friction_per_ps"] = args.friction_per_ps
    if args.log_interval is not None:
        config["equil_log_interval"] = args.log_interval
        config["prod_log_interval"] = args.log_interval
    if args.traj_interval is not None:
        config["equil_traj_interval"] = args.traj_interval
        config["prod_traj_interval"] = args.traj_interval
    if args.checkpoint_interval is not None:
        config["checkpoint_interval"] = args.checkpoint_interval

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    else:
        config["output_dir"] = _SCRIPT_DIR / ("md_ase_taper" if config["vdw_taper"] else "md_ase_notaper")

    if args.test:
        config.update(
            em_steps=50,
            nvt_steps=200,
            npt_steps=200,
            prod_steps=200,
            equil_traj_interval=50,
            equil_log_interval=20,
            prod_traj_interval=50,
            prod_log_interval=20,
            checkpoint_interval=50,
            output_dir=config["output_dir"].parent / (config["output_dir"].name + "_test"),
        )

    run_workflow(config)


if __name__ == "__main__":
    main()
