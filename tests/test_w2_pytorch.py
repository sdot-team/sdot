import pytest
import torch
import os
import sys

# Add build and python/pytorch to path
sys.path.append(os.path.join(os.getcwd(), 'python/pytorch'))
sys.path.append(os.path.join(os.getcwd(), 'build'))

from sdot_pytorch import sdot_w2

# def test_w2_pytorch_forward():
#     Xf = torch.tensor([0.25, 0.75], requires_grad=True)
#     Wf = torch.tensor([0.5, 0.5], requires_grad=True)
#     Xg = torch.tensor([0.0, 1.0])
#     Yg = torch.tensor([1.0, 1.0])

#     # Test distance only
#     dist = sdot_w2(Xf, Wf, Xg, Yg)
#     assert abs(dist.item()) < 1e-5

#     # Test distance and barycenters
#     dist, bary = sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=True)
#     assert torch.allclose(bary, torch.tensor([0.25, 0.75]))

# def test_w2_pytorch_backward():
#     Xf = torch.tensor([0.25, 0.75], requires_grad=True)
#     Wf = torch.tensor([0.5, 0.5], requires_grad=True)
#     Xg = torch.tensor([0.0, 1.0])
#     Yg = torch.tensor([1.0, 1.0])

#     dist = sdot_w2(Xf, Wf, Xg, Yg)
#     dist.backward()

#     # Backward stub returns zeros
#     assert torch.allclose(Xf.grad, torch.zeros_like(Xf))
