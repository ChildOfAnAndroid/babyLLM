from contextlib import nullcontext
from types import SimpleNamespace

import torch

from brain.LAYERS import tangling


def make_tangle(monkeypatch):
    monkeypatch.setattr(tangling, "embedDimension", 8)
    monkeypatch.setattr(tangling, "numNeurons", 16)
    counsellor = SimpleNamespace(infodump=lambda _: nullcontext())
    return tangling.MINI_INN_TANGLING(counsellor, 8, _device="cpu")


def test_repairs_collapsed_channels_without_reinitializing_healthy_stages(monkeypatch):
    mini = make_tangle(monkeypatch)
    assert mini.repair_collapsed_projections() == set()
    healthy_projection = mini.embed_proj_down.weight.detach().clone()
    with torch.no_grad():
        mini.neuron_proj_down.weight.fill_(0.01)
        mini.neuron_norm.bias.fill_(-2)
        mini.neuron_out_norm.bias.fill_(0.25)

    assert mini.repair_collapsed_projections() == {"neuron"}
    assert mini.repair_collapsed_projections() == set()
    assert mini.neuron_proj_down.weight.unique(dim=0).shape[0] == mini.MINI_NEURONS_LARGE
    assert mini.neuron_proj_up.weight.unique(dim=1).shape[1] == mini.MINI_NEURONS_LARGE
    torch.testing.assert_close(mini.neuron_norm.bias, torch.zeros(16))
    torch.testing.assert_close(mini.neuron_out_norm.bias, torch.full((16,), 0.25))
    torch.testing.assert_close(mini.embed_proj_down.weight, healthy_projection)


def test_chronology_update_needs_neuron_activity_and_trains_the_bottleneck(monkeypatch):
    mini = make_tangle(monkeypatch)
    neurons = torch.arange(1, 17, dtype=torch.float32).reshape(1, -1) * 0.1
    modulation = torch.full_like(neurons, 0.25, requires_grad=True)
    torch.testing.assert_close(mini.mix_chronology(neurons, torch.zeros_like(neurons)), neurons)
    torch.testing.assert_close(
        mini.mix_chronology(torch.zeros_like(neurons), modulation), torch.zeros_like(neurons)
    )

    output = mini.mix_chronology(neurons, modulation)
    assert not torch.allclose(output, neurons)
    output.square().sum().backward()
    assert modulation.grad.norm() > 0
    assert mini.neuron_proj_down.weight.grad.norm() > 0
    assert mini.neuron_proj_up.weight.grad.norm() > 0
    assert mini.neuron_norm.weight.grad is None
    assert mini.neuron_out_norm.weight.grad is None
