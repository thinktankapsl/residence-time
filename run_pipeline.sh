#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Automated Residence Time Pipeline
# =============================================================================
# Description:
#   End-to-end pipeline that takes GROMACS MD output files and computes the
#   ligand/ion residence time using PLUMED for CV calculation and Python for
#   survival-probability analysis.
#
# Steps:
#   1. Validate inputs
#   2. Center & image-correct the trajectory (gmx trjconv)
#   3. Run plumed driver to extract CVs
#   4. Run Python analysis to compute residence time
#   5. Print summary
#
# Requirements:
#   - GROMACS (gmx)
#   - PLUMED (plumed driver)
#   - Python ≥ 3.8 with: numpy, pandas, scipy, matplotlib
#
# Usage:
#   bash run_pipeline.sh [OPTIONS]
#
# Options:
#   -s  TPR file          (default: topol.tpr)
#   -f  Trajectory file   (default: traj_comp.xtc)
#   -p  PLUMED input      (default: plumed_residence.dat)
#   -c  Bound cutoff (nm) (default: 0.35)
#   -d  Timestep (ps)     (default: 0.002)
#   -o  Output directory  (default: results)
#   -h  Show this help
#
# Example:
#   bash run_pipeline.sh -s topol.tpr -f traj_comp.xtc -c 0.4 -d 0.002
# =============================================================================

set -euo pipefail

# ── Default parameters ────────────────────────────────────────────────────────
TPR="topol.tpr"
XTC="traj_comp.xtc"
PLUMED_IN="plumed_residence.dat"
CUTOFF="0.35"
TIMESTEP="0.002"
OUTDIR="results"
CENTERED_XTC="traj_centered.xtc"

# ── Colours for terminal output ────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse arguments ────────────────────────────────────────────────────────────
while getopts "s:f:p:c:d:o:h" opt; do
  case $opt in
    s) TPR="$OPTARG" ;;
    f) XTC="$OPTARG" ;;
    p) PLUMED_IN="$OPTARG" ;;
    c) CUTOFF="$OPTARG" ;;
    d) TIMESTEP="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    h) head -40 "$0" | grep "^#" | sed 's/^# \?//'; exit 0 ;;
    *) error "Unknown option -$OPTARG. Use -h for help." ;;
  esac
done

# ── Step 0: Validate dependencies and inputs ───────────────────────────────────
echo ""
info "=== Residence Time Pipeline ==="
info "Checking dependencies..."

for cmd in gmx plumed python3; do
  command -v $cmd &>/dev/null || error "$cmd not found. Please install it and re-run."
  info "  ✓ $cmd found"
done

for f in "$TPR" "$XTC" "$PLUMED_IN"; do
  [[ -f "$f" ]] || error "Required file not found: $f"
  info "  ✓ $f exists"
done

mkdir -p "$OUTDIR"
info "Output directory: $OUTDIR/"

# ── Step 1: Centre trajectory ──────────────────────────────────────────────────
info ""
info "=== Step 1/3: Centering trajectory with gmx trjconv ==="
info "Removing PBC artifacts and centring on Protein+Ligand..."

# The echo provides group selections interactively:
#   1st selection = group to centre on (Protein_LIG or 1 for Protein)
#   2nd selection = output group (System = everything)
# Adjust group numbers/names to match your system's index groups.
echo -e "Protein_LIG\nSystem" | gmx trjconv \
    -s "$TPR" \
    -f "$XTC" \
    -o "$CENTERED_XTC" \
    -center \
    -pbc mol \
    -ur compact \
    -quiet 2>&1 | tee "$OUTDIR/trjconv.log"

info "  ✓ Centred trajectory written to $CENTERED_XTC"

# ── Step 2: PLUMED driver ──────────────────────────────────────────────────────
info ""
info "=== Step 2/3: Running PLUMED driver to extract CVs ==="
info "PLUMED input : $PLUMED_IN"
info "Trajectory   : $CENTERED_XTC"
info "Timestep     : $TIMESTEP ps"

plumed driver \
    --plumed "$PLUMED_IN" \
    --mf_xtc "$CENTERED_XTC" \
    --timestep "$TIMESTEP" \
    --trajectory-stride 1 \
    2>&1 | tee "$OUTDIR/plumed.log"

[[ -f colvar.dat ]] || error "colvar.dat not created — check $OUTDIR/plumed.log for errors."
cp colvar.dat "$OUTDIR/colvar.dat"
info "  ✓ colvar.dat generated ($(wc -l < colvar.dat) frames)"

# ── Step 3: Python analysis ────────────────────────────────────────────────────
info ""
info "=== Step 3/3: Computing residence time (Python) ==="

python3 residence_time.py \
    --colvar colvar.dat \
    --cutoff "$CUTOFF" \
    --dt "$TIMESTEP" \
    --outdir "$OUTDIR"

# ── Final summary ─────────────────────────────────────────────────────────────
info ""
info "=== Pipeline complete ==="
info "Results saved to: $OUTDIR/"
info "  colvar.dat              — raw CV timeseries"
info "  residence_analysis.png  — timeseries + survival curve"
info "  residence_histogram.png — distribution of binding events"
info "  residence_summary.csv   — τ, bound fraction, statistics"
echo ""
