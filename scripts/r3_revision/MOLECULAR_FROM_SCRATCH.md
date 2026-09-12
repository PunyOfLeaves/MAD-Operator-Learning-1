# Molecular-domain reproduction from PQR

The small [7_GLY PQR](../../reproducibility/r3/inputs/7_GLY.pqr) is included.
It is the seven-atom input used for the experiment; its SHA-256 is recorded
in the input manifest. Only coordinates and radii are used. This is a
fixed-domain Laplace experiment, not a charged molecular PB/PNP calculation.

Install `requirements-r3.txt` and run the following commands from the MAD
repository root. Use fresh output directories. Large training arrays and
pretrained weights are unnecessary: the scripts produce both locally.

## 1. Build the geometric mesh

```powershell
python scripts/r3_revision/build_molecular_mesh.py `
  --pqr reproducibility/r3/inputs/7_GLY.pqr `
  --out data/r3_revision/molecular_fresh/mesh.msh `
  --report data/r3_revision/molecular_fresh/mesh_report.json `
  --padding_A 10 --h_interface_A 0.30 --h_outer_A 0.50 --h_bulk_A 1.0 `
  --dist_inner_A 0.40 --dist_outer_A 1.50 --radius_scale 1.0 `
  --order 1 --algorithm3d 10
```

This is the original sphere-union/OCC mesh construction, with its archived
size parameters. Gmsh fuses the atomic spheres, partitions an enclosing box,
and generates conforming tetrahedra. Physical volume 1 is the molecular
interior and physical surface 11 is its boundary. The enclosing solvent/box
is used only by the geometry builder; it is discarded for MAD training.
The meshing step does not solve a PDE or generate solution labels.

Meshes can differ with Gmsh/OCC versions and threading. A regenerated mesh
therefore reproduces the construction, not necessarily the archived node
ordering or exact polyhedral discretization. Do not claim bitwise replay of
the reported numbers on a remeshed domain. The archived full-mesh SHA-256 is
`58d6d42d5a93c217c6a895d5801dce4d12ed978351db8391a9fa402e4995c12a`.

## 2. Extract the domain and sample query points

```powershell
python scripts/r3_revision/prepare_molecular_geometry.py `
  --mesh data/r3_revision/molecular_fresh/mesh.msh `
  --output-dir data/r3_revision/molecular_fresh/geometry
```

The exporter verifies closed boundary/volume incidence, retains all boundary
nodes, and samples 2,048 training and 4,096 testing interior points by volume-
weighted tetrahedron selection and uniform barycentric coordinates. Seeds
2026091011 and 2026091036 match the original query-point protocol.

## 3. Generate MAD1 function samples

```powershell
python scripts/r3_revision/prepare_molecular_dataset.py `
  --geometry data/r3_revision/molecular_fresh/geometry/geometry.npz `
  --output-dir data/r3_revision/molecular_fresh/functions
```

Defaults are 2,000 training, 100 validation, and 100 independent test
functions. Each uses ten independently sampled exterior fundamental-solution
sources. Minimum clearance is 0.01 times the largest domain span, measured
against the actual polyhedral domain rather than just its bounding box.
The 50/30/20 source mixture includes exterior regions inside the box and
reentrant regions. The kernel is `1/(4*pi*|x-y|)`, with standard-normal
coefficients and joint boundary/interior maximum-amplitude normalization.

## 4. Train the paired models

```powershell
python scripts/r3_revision/train_molecular_paired.py `
  --data data/r3_revision/molecular_fresh/functions `
  --run artifacts/r3_revision/molecular_fresh/mad --method mad `
  --protocol reproducibility/r3/molecular_training.json
python scripts/r3_revision/train_molecular_paired.py `
  --data data/r3_revision/molecular_fresh/functions `
  --run artifacts/r3_revision/molecular_fresh/pi --method pi `
  --protocol reproducibility/r3/molecular_training.json
```

Both use the supplied width-220, 50,000-epoch configuration and identical
training boundary inputs. MAD uses boundary/interior labels; PI uses PDE
residuals and boundary labels without reading interior solution labels.
Each selects `best.pt` by its own validation objective, never by test error.

## 5. Evaluate fresh functions

```powershell
python scripts/r3_revision/evaluate_molecular_pair.py `
  --data data/r3_revision/molecular_fresh/functions `
  --mad-checkpoint artifacts/r3_revision/molecular_fresh/mad/best.pt `
  --pi-checkpoint artifacts/r3_revision/molecular_fresh/pi/best.pt `
  --output-dir artifacts/r3_revision/molecular_fresh/evaluation
```

The evaluator freezes both checkpoint hashes before generating 200 new
functions with seed 2026091107, checks that the models match the supplied
training inputs, and reports every case. Interior and boundary aggregate
relative L2 errors and per-function summaries are saved separately. These
are same-generator-law tests, not evidence of arbitrary-distribution or
unseen-geometry generalization. Never replace the archived paper numbers
with a subset selected from a new run.

`confirm_molecular_frozen.py` remains a separate exact-checkpoint audit and
intentionally rejects newly trained weights. Use the evaluator above for
fresh reproductions; do not remove the archived hash checks.
