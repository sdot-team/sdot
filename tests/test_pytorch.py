import pytest
import torch
import sys
import os

# Add build and python/pytorch to path
sys.path.append(os.path.join(os.getcwd(), 'python/pytorch'))
sys.path.append(os.path.join(os.getcwd(), 'build'))

# from sdot_pytorch import sdot_l2

def test_pouet():
    assert 1

# def test_sdot_l2_cpu():
#     f = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
#     g = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

#     loss = sdot_l2(f, g)

#     expected_loss = (0.0**2 + 1.0**2 + 2.0**2) / 3.0
#     assert abs(loss.item() - expected_loss) < 1e-6

#     loss.backward()

#     expected_grad_f = torch.tensor([0.0, 2/3, 4/3])
#     expected_grad_g = torch.tensor([0.0, -2/3, -4/3])

#     assert torch.allclose(f.grad, expected_grad_f)
#     assert torch.allclose(g.grad, expected_grad_g)

# def test_sdot_l2_mps():
#     if not torch.backends.mps.is_available():
#         pytest.skip("MPS not available")

#     f = torch.tensor([1.0, 2.0, 3.0], device='mps', requires_grad=True)
#     g = torch.tensor([1.0, 1.0, 1.0], device='mps', requires_grad=True)

#     loss = sdot_l2(f, g)
#     loss.backward()

#     assert abs(loss.item() - 1.6666666) < 1e-5
#     assert torch.allclose(f.grad.cpu(), torch.tensor([0.0, 2/3, 4/3]))
