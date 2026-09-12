"""Check paired loss, exact Laplacians, label access and restart behavior."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from molecular_laplace_experiment import write_json
from train_molecular_paired import load_fields, objective, train, trunk_laplacian
from train_molecular_laplace import MolecularDeepONet


class TinyOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_net = nn.Linear(4, 6, bias=False)
        self.trunk_net = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 6))
        self.last_layer_weights = nn.Parameter(torch.randn(6))


def repeated_ad_residual(model, boundary, coordinates):
    x = coordinates[None].repeat(len(boundary), 1, 1).detach().requires_grad_(True)
    coefficients = model.branch_net(boundary)*model.last_layer_weights
    values = (model.trunk_net(x)*coefficients[:, None, :]).sum(dim=-1)
    first = torch.autograd.grad(values.sum(), x, create_graph=True)[0]
    return sum(torch.autograd.grad(first[..., axis].sum(), x, create_graph=True)[0][..., axis]
               for axis in range(3))


class PairedChecks(unittest.TestCase):
    def test_residual_and_all_parameter_gradients_match_repeated_ad(self):
        torch.manual_seed(41)
        model = TinyOperator().double()
        boundary = torch.randn(2, 4, dtype=torch.float64)
        coordinates = torch.randn(7, 3, dtype=torch.float64)
        bc = torch.randn(4, 3, dtype=torch.float64)
        coefficients = model.branch_net(boundary)*model.last_layer_weights
        actual = coefficients @ trunk_laplacian(model.trunk_net, coordinates).T
        expected = repeated_ad_residual(model, boundary, coordinates)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        expected_loss = .9*expected.square().mean()+.1*((coefficients @ model.trunk_net(bc).T-boundary)**2).mean()
        actual_loss = objective(model, boundary, coordinates, bc, "pi")[0]
        a = torch.autograd.grad(actual_loss, tuple(model.parameters()))
        b = torch.autograd.grad(expected_loss, tuple(model.parameters()))
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-9)

    def test_affine_trunk_laplacian_is_zero(self):
        trunk = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 6)).double()
        result = trunk_laplacian(trunk, torch.randn(8, 3, dtype=torch.float64))
        torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)

    def test_full_width_float32_residual_and_gradients(self):
        torch.manual_seed(51)
        model = MolecularDeepONet(8)
        boundary, coordinates, bc = torch.randn(2, 8), torch.randn(4, 3), torch.randn(8, 3)
        expected = repeated_ad_residual(model, boundary, coordinates)
        coeff = model.branch_net(boundary)*model.last_layer_weights
        actual = coeff @ trunk_laplacian(model.trunk_net, coordinates).T
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=3e-4)
        direct_loss = .9*expected.square().mean()+.1*((coeff @ model.trunk_net(bc).T-boundary)**2).mean()
        loss = objective(model, boundary, coordinates, bc, "pi")[0]
        a = torch.autograd.grad(loss, tuple(model.parameters()))
        b = torch.autograd.grad(direct_loss, tuple(model.parameters()))
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, atol=2e-5, rtol=3e-4)

    def test_mad_is_mse_on_interior_and_boundary(self):
        torch.manual_seed(9)
        model = TinyOperator().double()
        boundary, truth = torch.randn(2, 4, dtype=torch.float64), torch.randn(2, 7, dtype=torch.float64)
        coords, bc = torch.randn(7, 3, dtype=torch.float64), torch.randn(4, 3, dtype=torch.float64)
        coeff = model.branch_net(boundary)*model.last_layer_weights
        direct = ((coeff @ model.trunk_net(torch.cat([coords, bc])).T-torch.cat([truth, boundary], dim=1))**2).mean()
        torch.testing.assert_close(objective(model, boundary, coords, bc, "mad", truth)[0], direct)

    def test_pi_does_not_read_interior_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            for split in ("train", "validation"):
                np.savez(data/(split+".npz"), boundary=np.zeros((2, 4), dtype="f4"),
                         solution=np.array([{"forbidden": "interior label"}], dtype=object))
            result = load_fields(data, "pi")
            self.assertEqual(set(result["train"]), {"boundary"})
            with self.assertRaises(ValueError):
                load_fields(data, "mad")

    def make_data(self, root):
        data = root/"data"
        data.mkdir()
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        points = np.array([[.1, .1, .1], [.2, .1, .1], [.1, .2, .1]])
        np.savez(data/"geometry.npz", nodes=nodes, boundary_ids=np.arange(4), train_queries=points,
                 center=np.zeros(3), scale=1.)
        rng = np.random.default_rng(21)
        for split, count in (("train", 4), ("validation", 2)):
            weights = rng.normal(size=(count, 3))
            offset = rng.normal(size=(count, 1))
            np.savez(data/(split+".npz"), boundary=(weights@nodes.T+offset).astype("f4"),
                     solution=(weights@points.T+offset).astype("f4"))
        write_json(data/"protocol.json", {"test": True})
        protocol = root/"paired.json"
        write_json(protocol, {"model_seed": 23, "batch_size": 2, "epochs": 4, "lr": 1e-4})
        return data, protocol

    def test_both_methods_resume_exactly_and_have_same_initialization(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            data, protocol = self.make_data(root)
            initials = []
            for method in ("mad", "pi"):
                full, split = root/(method+"_full"), root/(method+"_split")
                train(data, full, method, protocol)
                train(data, split, method, protocol, stop_after=2)
                train(data, split, method, protocol, resume=True)
                a = torch.load(full/"checkpoint.pt", weights_only=False)
                b = torch.load(split/"checkpoint.pt", weights_only=False)
                for key in a["model"]:
                    torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
                for first, second in zip(a["history"], b["history"]):
                    for key in first:
                        if key != "wall_seconds":
                            self.assertEqual(first[key], second[key])
                initials.append(torch.load(full/"initial.pt", weights_only=True))
                meta = json.loads((full/"run.json").read_text())
                self.assertEqual(meta["interior_training_and_validation_labels_loaded"], method == "mad")
            for key in initials[0]:
                torch.testing.assert_close(initials[0][key], initials[1][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
