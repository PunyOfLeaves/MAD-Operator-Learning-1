# PB reproduction without archived arrays or weights

Run from the MAD repository root. These PowerShell commands generate a new
MAD0 training array, train FNO, construct independent source-free tests,
and train/evaluate LNF-NO on the same new array. No pretrained weights or
large data download is required. Install `requirements-r3.txt` first.

This is an independent reproduction of the experimental procedure, not an
exact replay of the published numbers. The historical training-array random
state was not recorded. The seed 0 below defines the new data realization;
it is not asserted to regenerate the archived training array. Hardware and
software differences can also affect training trajectories. Both models
must use the same generated array and retain all 100 source-free tests.

## 1. Obtain the small upstream source files

The implementations are already public in
[LNF-NO at commit 2134b2](https://github.com/bzlu-Group/LNF-NO/tree/2134b24debeb5cb166dd3de558664c81a8d56c1c).
Fetch only the three relevant files, without checking out its data or models:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/bzlu-Group/LNF-NO.git external/LNF-NO
git -C external/LNF-NO checkout 2134b24debeb5cb166dd3de558664c81a8d56c1c -- data_gen/gen_pb_with_source.py scripts_train/FNO_PB_with_source.py scripts_train/LNFNO_PB_with_source.py
```

Public raw-file SHA-256 values (LF newlines):

| File | SHA-256 |
| --- | --- |
| `data_gen/gen_pb_with_source.py` | `67da1efbab04b8ff21f6ccf633d5a65930af443527d69db52c8b6861b0830147` |
| `scripts_train/FNO_PB_with_source.py` | `a0683f6d89ffc4acdfbd2f644e8cd269cd9c347bc326e9145a3b14871e9d2f20` |
| `scripts_train/LNFNO_PB_with_source.py` | `600318042df93bdc110b977eb9bd131f8f552edd8d77a634e679201f5190b76b` |

Git may use CRLF on Windows. The archived LNF-NO source checksum in the main
guide uses CRLF; the pinned public file is identical after newline normalization.

## 2. Generate new source-driven MAD0 data

Use a fresh clone/data destination. This uses the original 2-50-50-1 sine
generator, standard-normal parameters, random amplitude normalization,
float32 evaluation, and automatic differentiation for `Delta u - sinh(u) = f`.
The boundary ordering and `[g | f | u]` layout match the evaluation scripts.

```powershell
python -c "import sys, pathlib, numpy as np, torch; sys.path.insert(0,'external/LNF-NO/data_gen'); import gen_pb_with_source as g; torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0); p=pathlib.Path('external/LNF-NO/data/PB_WithSource_k1.0_2000_101.npy'); assert not p.exists(), 'Use a fresh destination'; a=g.generate_training_data_2d(2000,101,1.0,'cpu'); np.save(p,a)"
```

## 3. Train FNO from scratch

Use a new output directory so the upstream function cannot resume an old run.
Its `ntest=200` variable denotes the held-out source-driven validation split,
not the independent source-free test set. Normalization uses training rows
only. The retained checkpoint minimizes validation relative L2 error.

```powershell
python -c "import sys,pathlib,torch; sys.path.insert(0,'external/LNF-NO/scripts_train'); from FNO_PB_with_source import train_fno_pb_with_source as train; torch.set_num_threads(4); out=pathlib.Path('artifacts/r3_revision/pb_fresh/fno'); assert not out.exists(), 'Use a fresh run'; train(device='cuda' if torch.cuda.is_available() else 'cpu',data_dir='external/LNF-NO/data',save_dir=str(out),ntrain=1800,ntest=200,batch_size=32,epochs=500,lr=1e-3,weight_decay=1e-4,seed=0)"
```

## 4. Generate and evaluate all 100 source-free problems

```powershell
python scripts/r3_revision/evaluate_source_free_pb.py `
  --training-data external/LNF-NO/data/PB_WithSource_k1.0_2000_101.npy `
  --checkpoint artifacts/r3_revision/pb_fresh/fno/FNO_PB_k1.0_withsource_best.pt `
  --output-dir artifacts/r3_revision/pb_fresh/source_free `
  --samples 100 --seed 20260907 --boundary-family sine-network `
  --coarse-grid 201 --reference-grid 401
```

## 5. Train LNF-NO on the same array and compare

```powershell
python scripts/r3_revision/compare_lnfno_source_free_pb.py `
  --lnfno-script external/LNF-NO/scripts_train/LNFNO_PB_with_source.py `
  --training-data external/LNF-NO/data/PB_WithSource_k1.0_2000_101.npy `
  --fno-checkpoint artifacts/r3_revision/pb_fresh/fno/FNO_PB_k1.0_withsource_best.pt `
  --test-npz artifacts/r3_revision/pb_fresh/source_free/source_free_pb_sine_network_seed20260907_n100_grid401.npz `
  --test-json artifacts/r3_revision/pb_fresh/source_free/source_free_pb_sine_network_seed20260907_n100_grid401.json `
  --output-dir artifacts/r3_revision/pb_fresh/lnfno --epochs 500 --seed 0
```

The final command defaults to CUDA; add `--device cpu` if needed. It verifies
that both models use the same array using the new evaluation metadata and
records its checksum. Do not pass the archived array's checksum for a newly
generated array. The two configurations retain their respective learning
rates (FNO 1e-3 and LNF-NO 1e-4). Report all cases and the measured errors
from this run, not the original table values.
