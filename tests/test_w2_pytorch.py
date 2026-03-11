import torch
import sys
import os

# Add python/pytorch to path
sys.path.append(os.path.join(os.getcwd(), 'python/pytorch'))

from sdot_pytorch import sdot_w2

def test_w2_pytorch():
    print("Testing W2 PyTorch...")
    Xf = torch.tensor([0.25, 0.75], requires_grad=True)
    Wf = torch.tensor([0.5, 0.5], requires_grad=True)
    Xg = torch.tensor([0.0, 1.0])
    Yg = torch.tensor([1.0, 1.0])
    
    # Test distance only
    dist = sdot_w2(Xf, Wf, Xg, Yg)
    print(f"Dist: {dist.item()}")
    assert abs(dist.item()) < 1e-5
    
    # Test distance and barycenters
    dist, bary = sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=True)
    print(f"Barycenters: {bary}")
    assert torch.allclose(bary, torch.tensor([0.25, 0.75]))
    
    # Test backward stub
    dist.backward()
    print(f"Grad Xf: {Xf.grad}") # Should be zeros (stub)
    
    print("W2 PyTorch test passed!")

if __name__ == "__main__":
    test_w2_pytorch()
