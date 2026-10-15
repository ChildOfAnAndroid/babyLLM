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
                  and n.name == '_gradient_telemetry')
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
        holder = SimpleNamespace(named_parameters=lambda: iter(
            [(str(i), SimpleNamespace(grad=g)) for i,g in enumerate(gradients)] + [("unused", SimpleNamespace(grad=None))]))
        assert calculate(holder)["total"] == expected
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
    assert count == 0  # Debug reports consume shared scalar measurements.


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
    telemetry = norm_function(SOURCE.read_text())(holder, details=True)
    scope[node.name](holder, 5., 1.5, telemetry)
    output = capsys.readouterr().out
    assert 'failed safely' not in output
    assert 'CURRENT SENSORY STATE' in output
    assert output.index('invalid') < output.index('finite ')
    assert '5.000000' in output
    assert 'unused' not in output
    assert math.isnan(parameters[1][1].grad.item())
    torch.testing.assert_close(parameters[0][1].grad, torch.tensor([3.,4.]))


def test_shared_telemetry_matches_snapshot_and_clipping():
    measure = norm_function(SOURCE.read_text())
    ps = [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(3))]
    ps[0].grad = torch.tensor([1., 2., 0., -4.])
    ps[1].grad = torch.tensor([3., 0., -2.])
    holder = SimpleNamespace(named_parameters=lambda: iter(zip(['a','b'], ps)))
    result = measure(holder, details=True, snapshot_limit=1)
    assert result['maxima'] == {'a':4., 'b':3.}
    assert len(result['snapshot']) == 1
    stats = result['snapshot'][0][1]
    assert stats['norm'] == result['norms']['a']
    assert stats['shape'] == (4,)
    assert stats['sparsity'] == .25
    assert stats['mean'] == ps[0].grad.mean().item()
    assert stats['std'] == ps[0].grad.std(unbiased=False).item()
    norm = torch.nn.utils.clip_grad_norm_(ps, 1.5).item()
    assert math.isclose(result['total'], norm, rel_tol=1e-6)
    assert measure(holder)['snapshot'] == []
    assert measure(holder)['maxima'] == {}


def test_reporting_uses_optimizer_step_cadence():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == '_gradient_report_due')
    scope = {'gradientReportInterval': 3}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    holder = SimpleNamespace()
    due = scope[node.name]
    assert [due(holder) for _ in range(7)] == [True,False,False,True,False,False,True]
    scope['gradientReportInterval'] = 1
    assert all(due(holder) for _ in range(3))


def test_debug_statistics_include_absolute_mean_and_exact_count():
    measure = norm_function(SOURCE.read_text())
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.tensor([-3., 0., 1., 0.])
    holder = SimpleNamespace(named_parameters=lambda: iter([("embed.test", p)]))
    stats = dict(measure(holder, details=True, snapshot_limit=1)["snapshot"])["embed.test"]
    assert stats["abs_mean"] == 1.0
    assert stats["nonzero"] == 2
    assert stats["numel"] == 4
    assert stats["mean"] == -.5


def test_backward_debug_consumers_do_not_reduce_gradients_again():
    tree = ast.parse(SOURCE.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_backward_locked')
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != 'get_grad_stats'
    assert 'stats = before_stats[name]' in SOURCE.read_text()
    assert 'stats = after_stats[name]' in SOURCE.read_text()
