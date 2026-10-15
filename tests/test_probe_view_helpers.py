"""Probe cache contracts without Streamlit, GPU models or live checkpoints."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from utils.probe_view_helpers import (
    available_cache_files, load_probe_cache, load_three_vendor_sources,
    token_label, safe_category,
)


def test_vocab_and_sequence_cache_contracts(tmp_path):
    path = tmp_path / 'probe_seq_sample.npz'
    entries = [dict(tok='Ċ', labeled=False)]
    np.savez(path, entries_json=np.array([json.dumps(entries)]),
             neutral_embed=np.ones((1, 2)), tiled_embed=np.zeros((1, 2)),
             logit_lens_json=np.array([json.dumps([dict(token='hello')])]),
             blend_weights=np.ones((1, 4)))
    (tmp_path / 'probe_w32.npz').touch()
    assert len(available_cache_files(tmp_path)) == 1
    assert len(available_cache_files(tmp_path, include_sequences=True)) == 2
    assert available_cache_files(tmp_path / 'missing') == []
    loaded, neutral, tiled = load_probe_cache(path)
    assert loaded == entries
    np.testing.assert_array_equal(neutral['embed'], np.ones((1, 2)))
    np.testing.assert_array_equal(tiled['embed'], np.zeros((1, 2)))
    *_, extras = load_probe_cache(path, include_extras=True)
    assert extras['logit_lens'] == [dict(token='hello')]
    assert extras['blend_weights'].shape == (1, 4)
    assert token_label(entries[0]) == 'Ċ'
    assert token_label(dict(labeled=True, label='ʕっʘ‿ʘʔっ', tok='bpe')) == 'ʕっʘ‿ʘʔっ'
    assert safe_category({}) == 'unknown'


def test_vendor_sources_are_read_or_missing_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match='three.module.min.js'):
        load_three_vendor_sources(tmp_path)
    (tmp_path / 'three.module.min.js').write_text('three')
    (tmp_path / 'OrbitControls.js').write_text('orbit')
    assert load_three_vendor_sources(tmp_path) == ('three', 'orbit')


def test_profile_help_does_not_load_torch_or_model():
    root = Path(__file__).resolve().parents[1]
    code = '''import runpy, sys
sys.argv = ['profile_model.py', '--help']
try:
    runpy.run_path('tools/profile_model.py', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'babyLLM' not in sys.modules
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=root, check=True, capture_output=True)


@pytest.mark.parametrize('name', ['river', 'multitokenriver', 'theatre'])
def test_generated_viewer_javascript_parses(name, tmp_path):
    import ast
    import re
    import shutil
    from types import SimpleNamespace
    node_binary = shutil.which('node')
    if node_binary is None:
        pytest.skip('Node.js is needed for generated JavaScript syntax checks')
    source = Path(__file__).resolve().parents[1] / 'utils' / f'latent_probe_{name}.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_render_world')
    output = []
    namespace = dict(json=json, Dict=dict, Any=object,
                     components=SimpleNamespace(html=lambda html, **kw: output.append(html)),
                     _load_three_vendor_sources=lambda: ('export const example = 1;', 'export const example = 1;'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    namespace['_render_world']({})
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', output[0], re.S)
    assert scripts
    for index, script in enumerate(scripts):
        target = tmp_path / f'view-{index}.mjs'
        target.write_text(script)
        result = subprocess.run([node_binary, '--check', str(target)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
