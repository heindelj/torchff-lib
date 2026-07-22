#!/bin/bash
#SBATCH -A m2834
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J tip3p_taper_md
#SBATCH -o /pscratch/sd/e/eric6/torchff-lib/examples/tip3p/logs/tip3p_taper_md-%j.out
#SBATCH -e /pscratch/sd/e/eric6/torchff-lib/examples/tip3p/logs/tip3p_taper_md-%j.err
#
# TIP3P water_300 production MD with vdW taper ON (OpenMM switching + TorchFF taper).
#
# Submit from repo root:
#   sbatch examples/tip3p/sbatch_md_prod_taper.sh
#
# TorchFF backend (OpenMM driver):
#   sbatch --export=ALL,TORCHFF=1 examples/tip3p/sbatch_md_prod_taper.sh

set -euo pipefail

REPO_ROOT="/pscratch/sd/e/eric6/torchff-lib"
cd "$REPO_ROOT"

mkdir -p examples/tip3p/logs

module load conda
mamba activate openmm-torch-py312-cu124

TORCHFF="${TORCHFF:-0}"
TORCHFF_FLAG=(--openmm)
OUTPUT_DIR="examples/tip3p/md_openmm_taper"
if [[ "$TORCHFF" == "1" ]]; then
  TORCHFF_FLAG=(--torchff)
  OUTPUT_DIR="examples/tip3p/md_torchff_taper"
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Start: $(date)"
echo "Backend: $([[ "$TORCHFF" == 1 ]] && echo TorchFF || echo OpenMM)"
echo "vdw_taper: 1"
echo "Output dir: ${OUTPUT_DIR}"

python examples/tip3p/md_prod.py --vdw-taper "${TORCHFF_FLAG[@]}"

echo "End: $(date)"
echo "Done. Outputs in ${OUTPUT_DIR}/"
