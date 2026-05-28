# Ligand Residence Time Pipeline

Automated pipeline: GROMACS MD outputs → PLUMED collective variables → residence time (τ).

## Files

| File | Purpose |
|---|---|
| `run_pipeline.sh` | Master script — runs all three steps |
| `plumed_residence.dat` | PLUMED input — defines CVs |
| `residence_time.py` | Python analysis — survival fit |
| `get_atom_indices.py` | Helper — find atom indices |
| `setup_environment.sh` | Install Python dependencies |

## Quick Start

```bash
# 1. Install dependencies
bash setup_environment.sh

# 2. Edit plumed_residence.dat with your atom indices
python3 get_atom_indices.py --tpr topol.tpr --ligand LIG --site 45,67,89

# 3. Run the full pipeline
bash run_pipeline.sh -s topol.tpr -f traj_comp.xtc -c 0.35 -d 0.002
```

## Output

Results are saved to `results/`:
- `colvar.dat` — raw CV timeseries from PLUMED
- `residence_analysis.png` — CV trace, bound state indicator, survival curve with fits
- `residence_histogram.png` — distribution of binding episode durations
- `residence_summary.csv` — τ, bound fraction, fit parameters

## Key Parameters

| Parameter | Flag | Default | Notes |
|---|---|---|---|
| TPR file | `-s` | `topol.tpr` | GROMACS run input |
| Trajectory | `-f` | `traj_comp.xtc` | Compressed trajectory |
| Cutoff (nm) | `-c` | `0.35` | Bound/unbound threshold |
| Timestep (ps) | `-d` | `0.002` | Match your MD output stride |
| Output dir | `-o` | `results` | Where to save results |
