# R3 experiments

Additional experiments for *Mathematical artificial data for operator
learning*. The original scripts and data are unchanged. Run the commands
below from the repository root and use a new output directory for new runs.

## Experiment index and availability

Table numbers refer to the current R3 manuscript and Supplementary Materials.
The 50,000-epoch comparisons, rather than the diagnostic 5,000- or
200,000-epoch continuations, supply the reported training-data comparisons.

| Paper location | Experiment | Entry points in this directory |
| --- | --- | --- |
| Main Results, DST timing | Helmholtz, k=100, N=200 | `generate_helmholtz_datasets.py`, `benchmark_dst_accuracy.py` |
| Main Table 6; Table S.5 | MAD0 versus solver, three seeds | `train_original_protocol_vectorized.py`, `evaluate_original_protocol.py` |
| Main Table 6; Table S.6 | MAD1 versus solver, three seeds | `train_laplace_data_source_comparison.py`, `evaluate_laplace_data_source_comparison.py` |
| Main Fig. 6; Table S.7 | Fixed flower-shaped three-hole domain | `flower_holes_laplace_experiment.py`, `train_flower_holes_deeponet.py`, `evaluate_flower_holes_deeponet.py` |
| S4; Table S.4; Fig. S.26 | 100 source-free PB problems, FNO and LNF-NO | `evaluate_source_free_pb.py`, `compare_lnfno_source_free_pb.py`, `report_lnfno_source_free_pb.py` |
| S5.4; Table S.8; Fig. S.27 | Fixed 7_GLY domain, width-220 paired models | `train_molecular_paired.py`, `plot_7gly_boundary.py` |
| S5.4; Table S.9 | Frozen checkpoints, 200 fresh functions | `confirm_molecular_frozen.py` (requires the original checkpoints) |
| Additional sampling control | Same-input numerical relabeling | `relabel_mad0_solver.py` |

**Included inputs:** `reproducibility/r3/inputs/` contains the flower-domain
geometry/stiffness arrays and the exact held-out Laplace Test Set 2.
`manifest.json` records their provenance, sizes, and SHA-256 checksums.
`reproducibility/r3/results/` contains reviewed numerical summaries for the
three-seed comparisons, source-free PB comparison, and frozen molecular test.
Raw run folders and large arrays/checkpoints are not included. In particular,
the PB and molecular workflows require additional inputs listed in Section 8.

Install the base and optional geometry dependencies with
`python -m pip install -r requirements-r3.txt`. Mesh/refinement checks need
gmsh, meshio and scikit-fem; square-grid training/evaluation does not require
them. The training environments and mixed-hardware continuation history are
specified in the experimental protocols, not inferred from a current install.

## Completed 50,000-epoch comparisons

The R3 manuscript reports the completed 50,000-epoch paired comparisons:
MAD0 versus solver data for Helmholtz, and MAD1 versus solver data for
Laplace, each with seeds 20260904, 20260905, and 20260906 (12 runs).
Tables below summarize all three seeds using the original
minimum-training-minibatch-MSE checkpoints, not test-based selection.

The reported runs began on an RTX 4070 Laptop and continued on RTX 3090
GPUs, preserving model parameters, Adam state, minibatch-generator state,
and the running best checkpoint. Their cumulative timers must not be
interpreted as single-hardware training times. The continuation environment
used PyTorch 2.5.1+cu121 and NumPy 1.26.3.

The commands below specify the same total training budget for fresh paired
runs. Exact floating-point trajectories can depend on hardware and software.
The public summary is `reproducibility/r3/results/data_source_comparison.json`;
it retains all three seeds for each data source.

## 1. MAD0 versus solver-generated training data

Generate 2,000 samples for each data source for
`Delta u + u = f`. The solver inputs reproduce the fine-grid RBF-GP boundary
and Gaussian-filtered-noise source distribution used by Test Set 2.

```powershell
python scripts/r3_revision/generate_helmholtz_datasets.py `
  --samples 2000 --coefficient 1 --seed 20260904 `
  --output-dir data/r3_revision/formal_direct_gp
```

Train the same dual-branch DeepONet for 50,000 epochs on each dataset.
Repeat generation and training for all three paired seeds listed above:

```powershell
python scripts/r3_revision/train_original_protocol_vectorized.py `
  --data data/r3_revision/formal_direct_gp/mad0_k1_n2000_p51_seed20260904.npy `
  --output-dir artifacts/r3_revision/formal50000 `
  --run-name mad0_seed20260904_e50000 --epochs 50000 --seed 20260904

python scripts/r3_revision/train_original_protocol_vectorized.py `
  --data data/r3_revision/formal_direct_gp/dst_fd_k1_n2000_p51_seed20260904.npy `
  --output-dir artifacts/r3_revision/formal50000 `
  --run-name solver_seed20260904_e50000 --epochs 50000 --seed 20260904
```

The vectorized script preserves the original architecture, optimizer, batch
size, data order, loss, and minimum-training-batch checkpoint rule. It only
reuses branch features that the historical script redundantly evaluates at
every query coordinate. The two runs start from identical parameters and use
identical minibatch index sequences.

Evaluate both retained checkpoints on the two repository test sets and save
the reported metrics:

```powershell
python scripts/r3_revision/evaluate_original_protocol.py `
  --model MODEL_PATH --output EVALUATION_JSON_PATH
```

The revision reports means and sample standard deviations over the paired
seeds 20260904, 20260905, and 20260906. Within each seed pair, only the
training-data source changes. The aggregate relative L2 errors were:

| Training data | Test Set 1 | Test Set 2 |
| --- | ---: | ---: |
| MAD0 | 0.00732869 +/- 0.00067646 | 0.00590378 +/- 0.00023196 |
| DST-finite difference | 0.53260694 +/- 0.00452727 | 0.00071668 +/- 0.00006424 |

The corresponding generation times for 2,000 samples were
12.83 +/- 0.61 s for MAD0 and 157.38 +/- 3.57 s for the complete solver
pipeline (87.24 +/- 3.14 s for the DST solution step alone).

The formal comparison uses minimum-training-minibatch-MSE checkpointing,
not validation-based selection.

## 2. MAD1 versus GRF-boundary solver-generated training data

For the source-free Laplace comparison, generate 2,000 MAD1 fields and 2,000
finite-difference fields whose boundaries follow the original detrended
RBF-GP Test Set 2 distribution. Repeat the commands for seeds 20260904,
20260905, and 20260906.

```powershell
python scripts/r3_revision/generate_laplace_mad1_solver_datasets.py `
  --samples 2000 --seed 20260904 `
  --output-dir data/r3_revision/laplace_data_source/seed20260904

python scripts/r3_revision/train_laplace_data_source_comparison.py `
  --data DATASET_PATH --output-dir OUTPUT_DIR `
  --run-name RUN_NAME --epochs 50000 --seed 20260904

python scripts/r3_revision/evaluate_laplace_data_source_comparison.py `
  --model MODEL_PATH `
  --test-set-1 "data/MADlaplace2D1_(200, 51).txt" `
  --test-set-2 "reproducibility/r3/inputs/TSLlaplace2D_(200, 51).txt" `
  --output EVALUATION_JSON_PATH
```

The aggregate relative L2 errors were:

| Training data | MAD1 Test Set 1 | GRF-boundary Test Set 2 |
| --- | ---: | ---: |
| MAD1 fundamental-solution data | 0.00301933 +/- 0.00015481 | 0.00263225 +/- 0.00020978 |
| DST-finite-difference data | 0.04679448 +/- 0.00176750 | 0.00164447 +/- 0.00024088 |

Check the spatial convergence of the Laplace DST solver against four exact
harmonic modes with:

```powershell
python scripts/r3_revision/check_laplace_dst_accuracy.py
```

At grid spacing 0.001, the relative errors on the retained 51 by 51 grid
ranged from 4.81e-7 to 4.37e-5. Halving the grid spacing from 0.002 to 0.001
reduced the errors by approximately a factor of four.

## 3. Fast finite-difference timing

The following command compares float64 MAD0 generation with a five-point
finite-difference system diagonalized by a two-dimensional DST-I for 200
samples of `Delta u + 100 u = f` on a 1001 by 1001 grid:

```powershell
python scripts/r3_revision/generate_helmholtz_datasets.py `
  --samples 200 --coefficient 100 --workers 8 `
  --output-dir data/r3_revision/benchmark_k100
python scripts/r3_revision/benchmark_dst_accuracy.py
```

Report the DST solution time separately from the complete pipeline time, which
also includes random-field sampling, downsampling, normalization, and output.

## 4. Fixed flower-shaped three-hole domain

The reported complex-domain test uses a fixed smooth five-lobed outer boundary
with three rotated elliptical holes. The required P1 finite-element geometry
has 2,259 nodes, 4,042 triangles, and 480 boundary nodes. Generate 2,000 MAD1
fields, 200 independent MAD1 test fields, and 200 independent finite-element
test fields with Fourier-series data on all four boundary components:

```powershell
python scripts/r3_revision/flower_holes_laplace_experiment.py --stage generate
```

Train the reported five-layer tanh branch and trunk networks for 5,000 epochs.
The first 1,800 MAD1 fields are used for training and the remaining 200 for
validation and checkpoint selection:

```powershell
python scripts/r3_revision/train_flower_holes_deeponet.py
```

Evaluate the retained checkpoint on both independent test sets and regenerate
the mechanically selected median-error finite-element example:

```powershell
python scripts/r3_revision/evaluate_flower_holes_deeponet.py
```

The aggregate relative L2 errors were 0.0341495 on the independent MAD1 test
set and 0.0648069 on the independent finite-element test set. The corresponding
means and sample standard deviations of the per-sample errors were
0.0347780 +/- 0.0168497 and 0.0597146 +/- 0.0176007, respectively.

The following optional check regenerates a refined P1 mesh with mesh-size
parameter 0.01 and recomputes the first ten finite-element test fields:

```powershell
python scripts/r3_revision/check_flower_holes_fem_refinement.py
```

After interpolation to the coarse nodes, the mean and maximum relative L2
differences from the mesh generated with parameter 0.02 were 0.0029853 and
0.0051598. This test supports applicability on one more intricate fixed
geometry; it does not demonstrate transfer to unseen domains.

## 5. Two-dimensional source-free Poisson-Boltzmann evaluation

`evaluate_source_free_pb.py` evaluates the existing 101 by 101 MAD0-FNO
checkpoint without retraining on independently generated Dirichlet data for
`Delta u - sinh(u) = 0`. The default boundary family uses new randomized sine
networks with the same architecture, parameter distribution, and amplitude
normalization as the training generator, but the interior labels are computed
independently by a float64 finite-difference/DST Picard solver.

```powershell
python scripts/r3_revision/evaluate_source_free_pb.py `
  --training-data $TRAINING_DATA `
  --checkpoint $CHECKPOINT `
  --samples 100 --seed 20260907 `
  --coarse-grid 201 --reference-grid 401 `
  --boundary-family sine-network `
  --figure-path $FIGURE_PATH
```

For the reported 100-sample run, the mean, standard deviation, median, and
aggregate relative L2 errors were 0.09360, 0.03329, 0.09141, and 0.09442. The
mean and maximum 201-to-401 grid differences were 3.13e-5 and 6.84e-5. A
separate 10-sample 401-to-801 check gave mean and maximum differences of
7.53e-6 and 1.71e-5. The optional `fourier` boundary family is retained as an
internal distribution-shift diagnostic and is not used for the reported
source-free result.

## 6. LNF-NO on the same MAD0 training data

The architecture is described in H. Wu, J. Wang and B. Lu,
[Computer Modeling in Engineering & Sciences 148(2), 29 (2026)](https://doi.org/10.32604/cmes.2026.084608).
This experiment reuses its source-driven PB implementation, with the same
2,000-row array used for the original MAD0-FNO model: rows 0-1799 are training
data and rows 1800-1999 are validation data. No source-free labels enter
training, normalization, or checkpoint selection.

The required architecture file is `scripts_train/LNFNO_PB_with_source.py`
from the separate LNF-NO project. It is passed explicitly rather than
silently substituted with another implementation. Required checksums are
listed in Section 8.

After acquiring the required inputs listed below, the comparison entry point
is:

```powershell
python scripts/r3_revision/compare_lnfno_source_free_pb.py `
  --lnfno-script $LNFNO_SCRIPT --training-data $TRAINING_DATA `
  --fno-checkpoint $FNO_CHECKPOINT `
  --training-data-sha256 $TRAINING_DATA_SHA256 `
  --test-npz $SOURCE_FREE_NPZ --test-json $SOURCE_FREE_JSON `
  --output-dir $NEW_OUTPUT_DIR --epochs 500 --seed 0
```

Use the trusted original training-array checksum, not a checksum substituted
to accept a different array. Both methods use seed 0, 500 epochs, batch size
32, and source-driven validation selection. Each retains its original
optimization configuration (FNO learning rate 1e-3; LNF-NO 1e-4), so this
is a configuration comparison, not a single-factor architecture ablation.

| Metric on all 100 source-free problems | FNO | LNF-NO |
| --- | ---: | ---: |
| Mean full-grid relative L2 | 0.09360235 | 0.02321425 |
| Mean interior relative L2 | 0.08615497 | 0.02216845 |
| Mean boundary relative L2 | 0.15042986 | 0.03162589 |
| Aggregate relative L2 | 0.09442072 | 0.02371469 |

The representative figure retains the original FNO median-error example
(zero-based index 81); no samples are dropped from the statistics. The
public numerical summary is
`reproducibility/r3/results/source_free_pb_comparison.json`.

## 7. Final fixed 7_GLY comparison and fresh-function test

The final molecular-shaped experiment uses only the geometry, with
`Delta u = 0`; it is not a molecular electrostatics or PNP solve. It uses
the seven-atom glycine polyhedral interior, not the exploratory 3SGS domain.
The final source clearance is **0.01 times the largest bounding-box span**,
measured to the actual material domain, including concave exterior regions.
Use `prepare_molecular_dataset.py` for this final protocol; the lower-level
geometry exploration helpers are not substitutes for this entry point.

Both final models use the same 2,000 training and 100 validation functions,
bias-free linear boundary branch, four width-220 tanh trunk layers, seed
2026090906, batch size 32, Adam learning rate 1e-4, and 50,000 full-dataset
epochs on RTX 3090. The supervised method includes boundary and interior
solution values; the physics-constrained method does not read interior
solution labels for training or validation.

Given the original geometry (see Section 8), generate the shared data and
use the supplied common training configuration for both methods:

```powershell
python scripts/r3_revision/prepare_molecular_dataset.py `
  --geometry $GEOMETRY_NPZ --output-dir $FINAL_7GLY_DATA
python scripts/r3_revision/train_molecular_paired.py `
  --data $FINAL_7GLY_DATA --run $NEW_MAD_RUN `
  --method mad --protocol reproducibility/r3/molecular_training.json
python scripts/r3_revision/train_molecular_paired.py `
  --data $FINAL_7GLY_DATA --run $NEW_PI_RUN `
  --method pi --protocol reproducibility/r3/molecular_training.json
```

Table S.8 uses the two validation-selected checkpoints. Table S.9 is the
separate 200-function evaluation generated after freezing both checkpoints,
the sample count, seed 2026091107, and the generator. It retains the earlier
spatial query points. `confirm_molecular_frozen.py` deliberately requires the
archived checkpoint hashes and a fresh output directory; it is not a generic
evaluation command for arbitrarily retrained models. Do not bypass those
checks to call a new result an exact replay. The numerical summary is
`reproducibility/r3/results/molecular_frozen_comparison.json`. This test uses
exact exterior-source fields, not independently sampled FEM solutions.

## 8. Required inputs and release boundary

The following additional inputs are not included in this release. Exact
replay requires the original inputs; a documented replacement constitutes a
new experiment. Contact the authors for availability. Fresh retraining need
not reproduce an archived floating-point trajectory bit for bit.

| Input | Role / schema | Availability |
| --- | --- | --- |
| `1SourcePB2D_2000_101.npy` | PB source-driven (2000, 20802) array, fixed 1800/200 split | Not bundled |
| Original FNO best checkpoint | Frozen source-driven validation-selected model for S4 | Not bundled |
| `LNFNO_PB_with_source.py` | Separate LNF-NO architecture/training implementation | Not bundled; use the exact version identified below |
| Final 7_GLY `geometry.npz` | Geometry and fixed query points | Redistribution provenance pending confirmation; not bundled |
| Two frozen 7_GLY checkpoints | Exact Table S.9 replay | Not bundled; hashes enforced by the evaluation script |

Required SHA-256 values for the separate PB workflow:

- Training array: `1757d932233d7f385639d7b042f4b451df33b3b412f9b6ab9dbc255ab50b400b`.
- LNF-NO source file: `01e0f34688c3a2b01b4f068cb01e725b9e298684fb0efd087f619df2eedb291b`.
- Final molecular geometry: `4defaf2e5931424be0121a96ecae570f2d0642a697c4225d73cb863429f96dc2`.

The molecular geometry must supply `nodes`, `tetra`, `boundary_ids`,
`train_queries`, `test_queries`, `center`, and `scale` arrays.
Its original coordinates and scaling must be retained for exact replay.

The same-input relabeling control is a single-seed diagnostic, not a fourth
seed or part of Main Table 6. `relabel_mad0_solver.py --help` describes its
inputs. Do not merge its statistics with the three-seed comparisons.

Run the lightweight checks without training the reported experiments:

```powershell
python -m unittest discover -s scripts/r3_revision -p "test_*.py"
```

The supplied result summaries are evidence records, not substitutes for the
unbundled arrays or weights needed to replay the PB and molecular results.
