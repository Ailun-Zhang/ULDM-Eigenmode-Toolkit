#!/bin/bash
# SLURM template: one PyUL run -> a_lm checkpoints -> c_nlm.h5 with the one-pass pipeline.
# Submit with:  sbatch run_cnlm_pipeline.slurm     (edit the block below first)
# Re-submitting the same job resumes from Outputs/cnlm_pipeline/alm_checkpoints/;
# finished outputs are never overwritten unless you add --force.
#SBATCH --job-name=uldm_cnlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=uldm_cnlm_%j.out

set -euo pipefail

# ---------------- edit these ----------------
RUN_DIR="/path/to/save_path/YYYYMMDD_HHMMSS@256"   # PyUL run directory (contains config.uldm)
PYTHON="python3"                                    # interpreter of the environment with Eig_ULDM_packages
# Extra pipeline options. For PyUL <= 30.40 runs the resampled box must be given, e.g.
#   --rs-side 5.0 --rs-resol 256   (kpc; with CellCenteredGrid=True 30.38-30.40 used (L - L/N)*Length_Ratio)
PIPELINE_ARGS=(--build-eigen)
# Optional environment activation (left disabled; set -u is relaxed while it runs)
ACTIVATE=""                                         # e.g. "module load python; source /path/to/venv/bin/activate"
# --------------------------------------------

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -n "${ACTIVATE}" ]; then
    set +u
    eval "${ACTIVATE}"
    set -u
fi

# One process, ULDM_NUM_WORKERS threads; keep BLAS/OpenMP single-threaded to avoid oversubscription.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ULDM_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

LOG="cnlm_pipeline_${SLURM_JOB_ID:-nojob}_$(date +%Y%m%d_%H%M%S).log"
echo "node: $(hostname)  job: ${SLURM_JOB_ID:-none}  workers: ${ULDM_NUM_WORKERS}  run: ${RUN_DIR}" | tee "${LOG}"
"${PYTHON}" -u -m Eig_ULDM_packages.pipeline "${RUN_DIR}" "${PIPELINE_ARGS[@]}" 2>&1 | tee -a "${LOG}"
echo "finished: $(date -Is)" | tee -a "${LOG}"
