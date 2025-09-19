# v1.7
import importlib
import sys
import threading
import types
from pathlib import Path

import torch
import torch.nn as nn


class DummyContext:
    def __enter__(self):
        return lambda *args, **kwargs: None

    def __exit__(self, exc_type, exc, tb):
        return False


class EmptyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _install_stub_modules(monkeypatch):
    secret = types.ModuleType("secret")
    secret.tensor_snitch = lambda *args, **kwargs: None
    secret.model_thread_lock = threading.Lock()
    monkeypatch.setitem(sys.modules, "secret", secret)

    sophia_pkg = types.ModuleType("sophia")
    sophia_pkg.__path__ = []  # mark as package
    sophia_sub = types.ModuleType("sophia.sophia")

    class DummySophia:
        def __init__(self, *args, **kwargs):
            pass

    sophia_sub.SophiaG = DummySophia
    sophia_pkg.sophia = sophia_sub
    monkeypatch.setitem(sys.modules, "sophia", sophia_pkg)
    monkeypatch.setitem(sys.modules, "sophia.sophia", sophia_sub)

    adan_pkg = types.ModuleType("adan_pytorch")

    class DummyAdan:
        def __init__(self, params, **kwargs):
            self.params = list(params)
            self.kwargs = kwargs

    adan_pkg.Adan = DummyAdan
    monkeypatch.setitem(sys.modules, "adan_pytorch", adan_pkg)

    brain_pkg = types.ModuleType("brain")
    brain_pkg.__path__ = []
    layers_pkg = types.ModuleType("brain.LAYERS")
    layers_pkg.__path__ = []

    embed_mod = types.ModuleType("brain.LAYERS.embed")
    embed_mod.EMBED = EmptyModule
    inn_mod = types.ModuleType("brain.LAYERS.interneuronNetwork")
    inn_mod.INTERNEURON_NETWORK = EmptyModule
    logits_mod = types.ModuleType("brain.LAYERS.logits")
    logits_mod.LOGITS = EmptyModule
    memory_mod = types.ModuleType("brain.LAYERS.memory")
    memory_mod.MEMORY = EmptyModule
    attn_mod = types.ModuleType("brain.LAYERS.attention")
    attn_mod.GATED_MHA = EmptyModule

    monkeypatch.setitem(sys.modules, "brain", brain_pkg)
    monkeypatch.setitem(sys.modules, "brain.LAYERS", layers_pkg)
    monkeypatch.setitem(sys.modules, "brain.LAYERS.embed", embed_mod)
    monkeypatch.setitem(sys.modules, "brain.LAYERS.interneuronNetwork", inn_mod)
    monkeypatch.setitem(sys.modules, "brain.LAYERS.logits", logits_mod)
    monkeypatch.setitem(sys.modules, "brain.LAYERS.memory", memory_mod)
    monkeypatch.setitem(sys.modules, "brain.LAYERS.attention", attn_mod)


class DummyCounsellor:
    def infodump(self, *_args, **_kwargs):
        return DummyContext()


class DummyLibrarian:
    vocabList = list(range(8))


def test_adan_optimizer_selection(monkeypatch):
    _install_stub_modules(monkeypatch)

    for name in list(sys.modules):
        if name.startswith("babyLLM") or name.startswith("BRAIN"):
            sys.modules.pop(name, None)

    importlib.invalidate_caches()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    babyLLM = importlib.import_module("babyLLM")

    monkeypatch.setattr(babyLLM, "optimizerName", "Adan", raising=False)
    monkeypatch.setattr(babyLLM, "EMBED", EmptyModule, raising=False)
    monkeypatch.setattr(babyLLM, "GATED_MHA", EmptyModule, raising=False)
    monkeypatch.setattr(babyLLM, "INTERNEURON_NETWORK", EmptyModule, raising=False)
    monkeypatch.setattr(babyLLM, "LOGITS", EmptyModule, raising=False)
    monkeypatch.setattr(babyLLM, "MEMORY", EmptyModule, raising=False)

    model = babyLLM.BABYLLM(
        DummyCounsellor(),
        _calligraphist=None,
        _scribe=None,
        _librarian=DummyLibrarian(),
        _numTokensPerStep=4,
        _device=torch.device("cpu"),
    )

    assert isinstance(model.optimizer, babyLLM.Adan)
    assert model.optimizer.kwargs["lr"] == babyLLM.learningRate
