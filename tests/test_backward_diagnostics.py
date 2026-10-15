"""Exercise diagnostic math without importing or starting the live model."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'babyLLM.py'


def norm_function(source):
    tree = ast.parse(source)
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_excite_current_grad_norm')
    method.decorator_list = []
    namespace = {'torch': torch, 'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[method.name]


def test_gradient_norm_preserves_values_and_gradients():
    calculate = norm_function(SOURCE.read_text())
    for values, expected in [([], 0.), ([3., 4.], 5.),
                             ([float('nan')], float('inf')),
                             ([float('inf')], float('inf'))]:
        gradients = [torch.tensor([v]) for v in values]
        before = [g.clone() for g in gradients]
        holder = SimpleNamespace(parameters=lambda: iter(
            [SimpleNamespace(grad=g) for g in gradients] + [SimpleNamespace(grad=None)]))
        assert calculate(holder) == expected
        for a, b in zip(gradients, before):
            torch.testing.assert_close(a, b, equal_nan=True)


def test_disabled_debug_norms_are_not_evaluated():
    tree = ast.parse(SOURCE.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_backward_locked')
    parents = {child: node for node in ast.walk(method) for child in ast.iter_child_nodes(node)}
    count = 0
    for node in ast.walk(method):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'debug_print'
                and any(isinstance(x, ast.Attribute) and x.attr == 'norm' for x in ast.walk(node))):
            continue
        count += 1
        parent = node
        while parent in parents:
            parent = parents[parent]
            if isinstance(parent, ast.If) and isinstance(parent.test, ast.Name) and parent.test.id == 'debugPrints':
                break
        else:
            raise AssertionError('eager diagnostic norm outside debug guard')
    assert count == 4
