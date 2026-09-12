"""Predeclared fresh-function evaluation of two frozen 7GLY checkpoints."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
import torch
from molecular_laplace_experiment import ROOT, digest, write_json
from molecular_geometry_sources import MolecularGeometry, triangle_distances
from molecular_laplace_true_exterior import generate_samples
from train_molecular_laplace import MolecularDeepONet


def main(args):
    base = args.data.resolve()
    out = args.output_dir.resolve()
    checkpoints = {
        'mad': args.mad_checkpoint.resolve(),
        'pi': args.pi_checkpoint.resolve()}
    expected = {'mad': '911328a8d7ae9f7ebb677f22175a346e3cfd61ca2a26e83d0eb896998e922712',
                'pi': '8c35ea8be4ec17329dde539b34cb3509e343c43579c154d9df3d66bc8dd07b71'}
    for name, path in checkpoints.items():
        assert digest(path) == expected[name]
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), count=200,
                    seed=2026091107, relative_clearance=0.01,
                    checkpoints=expected, geometry_sha256=digest(base/'geometry.npz'),
                    source_protocol_sha256=digest(base/'protocol.json'),
                    script_sha256=digest(Path(__file__)),
                    scope='Fresh functions, fixed existing spatial queries and generator law; no training or selection. Report all 200 irrespective of results.',
                    metrics='Separate interior/boundary aggregate relative Frobenius error; per-function mean, sample SD, median, p90, maximum.')
    write_json(out/'protocol_before_generation.json', protocol)
    geo = np.load(base/'geometry.npz')
    geometry = MolecularGeometry(geo['nodes'], geo['tetra'])
    boundary = geo['nodes'][geo['boundary_ids']]
    a, stats = generate_samples(geometry, boundary, geo['test_queries'], 200, 2026091107, relative_clearance=.01)
    assert all(np.isfinite(v).all() for v in a.values())
    assert np.all(a['source_clearance'] > .01*geometry.scale)
    for p in a['sources'].reshape(-1, 3)[::20]:
        assert not geometry.contains(p)
        assert triangle_distances(p, geometry.triangles).min() > .01*geometry.scale
    for split in ['train', 'validation', 'mad_test']:
        old = np.load(base/(split+'.npz'))['sources']
        assert not ({x.tobytes() for x in old} & {x.tobytes() for x in a['sources']})
    np.savez(out/'fresh_test.npz', **a)
    write_json(out/'generation.json', dict(rejections=stats, test_sha256=digest(out/'fresh_test.npz'), minimum_clearance=float(a['source_clearance'].min())))
    print('Fresh test generated and verified; starting frozen evaluation.', flush=True)
    torch.set_num_threads(2)
    report = {}
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        m = saved['metadata']
        model = MolecularDeepONet(m['boundary_count'], m['hidden_width'])
        model.load_state_dict(saved['model'])
        model.eval()
        result = {'epoch': saved['epoch'], 'checkpoint_sha256': digest(path)}
        for region, points, key in [('interior', geo['test_queries'], 'solution'), ('boundary', boundary, 'boundary')]:
            q = torch.tensor((points-geo['center'])/geo['scale'], dtype=torch.float32)
            predictions = []
            with torch.inference_mode():
                for start in range(0, 200, 32):
                    x = torch.tensor((a['boundary'][start:start+32]-m['x_mean'])/(m['x_std']+m['normalizer_eps']), dtype=torch.float32)
                    predictions.append((model(x,q)*(m['y_std']+m['normalizer_eps'])+m['y_mean']).numpy())
            pred = np.concatenate(predictions).astype(np.float64)
            truth = a[key].astype(np.float64)
            errors = np.linalg.norm(pred-truth, axis=1)/np.linalg.norm(truth, axis=1)
            result[region] = dict(count=200, aggregate=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)), mean=float(errors.mean()), sd=float(errors.std(ddof=1)), median=float(np.median(errors)), p90=float(np.quantile(errors,.9)), maximum=float(errors.max()))
            np.savez(out/(name+'_'+region+'.npz'), prediction=pred, per_function_error=errors)
        assert digest(path) == expected[name]
        report[name] = result
    write_json(out/'results.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mad-checkpoint', type=Path, required=True)
    parser.add_argument('--pi-checkpoint', type=Path, required=True)
    main(parser.parse_args())
