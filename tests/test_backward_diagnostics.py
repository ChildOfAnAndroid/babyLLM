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


def test_microscope_preserves_ranking_and_nonfinite_reporting(capsys):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == '_print_excitement_report')
    scope = {'torch': torch, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    parameters = []
    for name, values in [('finite', [3., 4.]), ('invalid', [float('nan')]), ('unused', None)]:
        parameter = torch.nn.Parameter(torch.zeros(len(values) if values else 1))
        parameter.grad = torch.tensor(values) if values else None
        parameters.append((name, parameter))
    holder = SimpleNamespace(named_parameters=lambda: iter(parameters),
        inputBlend=SimpleNamespace(grad=None), charBlendWeight=SimpleNamespace(grad=None),
        _excite_last_blend=None, _excite_token_losses=[])
    scope[node.name](holder, 5., 1.5)
    output = capsys.readouterr().out
    assert 'failed safely' not in output
    assert 'CURRENT SENSORY STATE' in output
    assert output.index('invalid') < output.index('finite ')
    assert '5.000000' in output
    assert 'unused' not in output
    assert math.isnan(parameters[1][1].grad.item())
    torch.testing.assert_close(parameters[0][1].grad, torch.tensor([3.,4.]))
