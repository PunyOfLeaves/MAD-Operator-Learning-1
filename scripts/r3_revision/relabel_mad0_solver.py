"""Frozen-input DST relabeling; never rescale the original input tuples."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from generate_helmholtz_datasets import (
    FastHelmholtzSolver, SineGenerator, set_seed, square_boundary, square_grid,
)


def fields(model, points, coefficient=1.0):
    """Analytic chain rule for the existing two-sine-layer generator."""
    w1, w2, w3 = model.fc1.weight, model.fc2.weight, model.fc3.weight
    z1 = model.fc1(points)
    a1 = torch.sin(z1)
    z2 = model.fc2(a1)
    u = model.fc3(torch.sin(z2))
    lap = torch.zeros_like(u)
    for axis in range(2):
        d1 = torch.cos(z1) * w1[:, axis]
        dd1 = -a1 * w1[:, axis].square()
        d2 = d1 @ w2.T
        dd2 = dd1 @ w2.T
        lap += (-torch.sin(z2) * d2.square() + torch.cos(z2) * dd2) @ w3.T
    return u[:, 0], (lap + coefficient * u)[:, 0]


def autograd_reference(model, points):
    points = points.clone().requires_grad_(True)
    u = model(points)
    gradient = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    lap = sum(torch.autograd.grad(gradient[:, d].sum(), points,
                                 retain_graph=True)[0][:, d] for d in range(2))
    return u.detach()[:, 0], (lap + u[:, 0]).detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--grids', type=int, nargs='+', default=[501, 1001])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original = np.load(args.original, mmap_mode='r')
    if original.shape[1] != 5402 or not 1 <= args.samples <= len(original):
        raise ValueError('Expected original 51x51 Helmholtz tuples.')
    if any((n-1) % 50 or n < 51 for n in args.grids):
        raise ValueError('All fine grids must contain the original queries.')
    protocol = dict(seed=args.seed, samples=args.samples, grids=args.grids,
                    original=str(args.original.resolve()),
                    original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    coefficient=1.0, normalization='original MAD scale; inputs copied exactly',
                    training_epochs=50000, training_N=2000,
                    purpose='same-input label-source control, not independent input-sampler design')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    coarse = square_grid(51, torch.float64)
    boundary = square_boundary(51, torch.float64)
    solvers = {n: FastHelmholtzSolver(n, 1.0, args.threads) for n in args.grids}
    arrays = {n: np.lib.format.open_memmap(args.output/f'data_p{n}.npy', mode='w+',
               dtype=np.float64, shape=(args.samples, 5402)) for n in args.grids}
    results = []
    start = time.perf_counter()
    for i in range(args.samples):
        model = SineGenerator().eval()
        _, reference_f = autograd_reference(model, coarse[:97])
        with torch.no_grad():
            u, f = fields(model, coarse)
            np.testing.assert_allclose(f[:97].numpy(), reference_f.numpy(), rtol=1e-11, atol=1e-10)
            g = model(boundary)[:, 0]
            raw = torch.cat((g, f, u)).numpy()
            scale = np.max(np.abs(raw))
            np.testing.assert_allclose(raw/scale, original[i], rtol=2e-10, atol=2e-12)
            record = {'index': i, 'grids': {}}
            for n, solver in solvers.items():
                tick = time.perf_counter()
                fine_source = np.empty(n*n, dtype=np.float64)
                # Chunk coordinates and hidden activations to keep memory bounded.
                for first in range(0, n*n, 8192):
                    indices = torch.arange(first, min(first+8192, n*n))
                    xy = torch.stack((indices % n, indices // n), dim=1).double()/(n-1)
                    _, source = fields(model, xy)
                    fine_source[first:first+len(indices)] = source.numpy()/scale
                fine_g = model(square_boundary(n, torch.float64))[:, 0].numpy()/scale
                numerical = solver.solve(fine_g[None], fine_source.reshape(1,n,n))[0]
                sampled = numerical[::((n-1)//50), ::((n-1)//50)].ravel()
                arrays[n][i, :2801] = original[i, :2801]
                arrays[n][i, 2801:] = sampled
                truth = original[i, 2801:]
                record['grids'][str(n)] = dict(
                    relative_l2=float(np.linalg.norm(sampled-truth)/np.linalg.norm(truth)),
                    max_absolute_error=float(np.max(np.abs(sampled-truth))),
                    seconds=time.perf_counter()-tick)
                arrays[n].flush()
        results.append(record)
        (args.output/'results.json').write_text(json.dumps(
            dict(completed=len(results), elapsed_seconds=time.perf_counter()-start,
                 samples=results), indent=2))
        print(json.dumps(record), flush=True)
    for n, arr in arrays.items():
        assert np.array_equal(arr[:, :2801], original[:args.samples, :2801])
    print('Completed; original inputs preserved exactly.', flush=True)


if __name__ == '__main__':
    main()
