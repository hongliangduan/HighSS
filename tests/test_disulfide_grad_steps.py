"""Toy check of the opt-in gradient path through the last K diffusion steps.

Run in the HighSS environment (needs torch + the vendored boltz package):
    python -m pytest tests/test_disulfide_grad_steps.py -q
It uses a fake denoiser, not the real model, so it only tests the sampler logic:
  * K = 0 -> output carries no graph, every network call runs under no_grad (RC3 behaviour)
  * K = 2 -> exactly the last two network calls run with autograd and without the cache,
             and the gradient reaches the (stand-in) conditioning tensor W.
"""
import pytest

torch = pytest.importorskip("torch")
from boltz.model.modules.diffusion import AtomDiffusion  # noqa: E402


class _FakeDiffusion:
    sample = AtomDiffusion.sample
    sample_schedule = AtomDiffusion.sample_schedule

    def __init__(self, alignment_reverse_diff):
        self.device = torch.device("cpu")
        self.sigma_min, self.sigma_max, self.sigma_data, self.rho = 0.0004, 160.0, 16.0, 7
        self.num_sampling_steps = 8
        self.gamma_0, self.gamma_min = 0.605, 1.107
        self.noise_scale, self.step_scale = 0.901, 1.638
        self.alignment_reverse_diff = alignment_reverse_diff
        self.use_inference_model_cache = True
        self.accumulate_token_repr = False
        self.W = torch.randn(1, 6, 3, requires_grad=True)
        self.calls = []

    def preconditioned_network_forward(self, x, t, training, network_condition_kwargs):
        self.calls.append((torch.is_grad_enabled(), network_condition_kwargs["model_cache"] is None))
        return 0.5 * x + self.W, None


@pytest.mark.parametrize("align", [False, True])
def test_no_grad_by_default(align):
    d = _FakeDiffusion(align)
    out = d.sample(atom_mask=torch.ones(1, 6), grad_last_steps=0)["sample_atom_coords"]
    assert not out.requires_grad
    assert all(not g for g, _ in d.calls)


@pytest.mark.parametrize("align", [False, True])
def test_grad_through_last_steps(align):
    d = _FakeDiffusion(align)
    out = d.sample(atom_mask=torch.ones(1, 6), grad_last_steps=2)["sample_atom_coords"]
    assert out.requires_grad
    grads = [g for g, _ in d.calls]
    assert grads == [False] * (len(grads) - 2) + [True, True]
    assert all(no_cache for g, no_cache in d.calls if g)
    out.pow(2).sum().backward()
    assert d.W.grad is not None and torch.isfinite(d.W.grad).all() and d.W.grad.abs().sum() > 0
