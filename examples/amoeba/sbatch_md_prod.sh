#!/bin/bash
#SBATCH -A m2834
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J amoeba_md_prod
#SBATCH -o /pscratch/sd/e/eric6/torchff-lib/examples/amoeba/logs/amoeba_md_prod-%j.out
#SBATCH -e /pscratch/sd/e/eric6/torchff-lib/examples/amoeba/logs/amoeba_md_prod-%j.err
#
# AMOEBA water_300 production MD (minimization + 500 ps NVT + 500 ps NPT + 5 ns NPT).
#
# Submit from repo root:
#   sbatch examples/amoeba/sbatch_md_prod.sh
#
# TorchFF backend:
#   sbatch --export=ALL,TORCHFF=1,VDW_TAPER=1 examples/amoeba/sbatch_md_prod.sh
#   sbatch --export=ALL,TORCHFF=1,VDW_TAPER=0 examples/amoeba/sbatch_md_prod.sh  # -> md_torchff_notaper

set -euo pipefail

REPO_ROOT="/pscratch/sd/e/eric6/torchff-lib"
cd "$REPO_ROOT"

mkdir -p examples/amoeba/logs

module load conda
mamba activate openmm-torch-py312-cu124

TORCHFF="${TORCHFF:-0}"
VDW_TAPER="${VDW_TAPER:-0}"
TORCHFF_FLAG=()
OUTPUT_DIR="examples/amoeba/md_openmm"
if [[ "$TORCHFF" == "1" ]]; then
  TORCHFF_FLAG=(--torchff)
  if [[ "$VDW_TAPER" == "1" ]]; then
    TORCHFF_FLAG+=(--vdw-taper)
    OUTPUT_DIR="examples/amoeba/md_torchff_taper"
  else
    TORCHFF_FLAG+=(--no-vdw-taper)
    OUTPUT_DIR="examples/amoeba/md_torchff_notaper"
  fi
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start: $(date)"
echo "Backend: $([[ "$TORCHFF" == 1 ]] && echo TorchFF || echo OpenMM)"
echo "vdw_taper: ${VDW_TAPER}"
echo "Output dir: ${OUTPUT_DIR}"

python examples/amoeba/md_prod.py "${TORCHFF_FLAG[@]}"

echo "End: $(date)"
echo "Done. Outputs in ${OUTPUT_DIR}/"
