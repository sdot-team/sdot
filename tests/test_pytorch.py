import torch
import sys
import os

# Add python/pytorch to path
sys.path.append(os.path.join(os.getcwd(), 'python/pytorch'))

from sdot_pytorch import sdot_l2

def test_sdot_l2_cpu():
    print("Testing CPU...")
    f = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    g = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    
    loss = sdot_l2(f, g)
    print(f"Loss: {loss.item()}")
    
    expected_loss = (0.0**2 + 1.0**2 + 2.0**2) / 3.0
    assert abs(loss.item() - expected_loss) < 1e-6
    
    loss.backward()
    
    print(f"Grad F: {f.grad}")
    print(f"Grad G: {g.grad}")
    
    expected_grad_f = torch.tensor([0.0, 2/3, 4/3])
    expected_grad_g = torch.tensor([0.0, -2/3, -4/3])
    
    assert torch.allclose(f.grad, expected_grad_f)
    assert torch.allclose(g.grad, expected_grad_g)
    print("CPU test passed!")

def test_sdot_l2_mps():
    if not torch.backends.mps.is_available():
        print("MPS not available, skipping.")
        return
    
    print("Testing MPS...")
    # Note: Our current implementation for MPS copies to CPU internally
    f = torch.tensor([1.0, 2.0, 3.0], device='mps', requires_grad=True)
    g = torch.tensor([1.0, 1.0, 1.0], device='mps', requires_grad=True)
    
    loss = sdot_l2(f, g)
    print(f"Loss (MPS): {loss.item()}")
    
    loss.backward()
    print(f"Grad F (MPS): {f.grad}")
    
    assert abs(loss.item() - 1.6666666) < 1e-5
    # Move grad back to cpu to check
    assert torch.allclose(f.grad.cpu(), torch.tensor([0.0, 2/3, 4/3]))
    print("MPS test passed!")

if __name__ == "__main__":
    # Ensure kernel file is in current dir or predictable path
    if not os.path.exists('sdot_l2_kernels.metal'):
        import shutil
        shutil.copy('src/metal/sdot_l2_kernels.metal', '.')
        
    test_sdot_l2_cpu()
    test_sdot_l2_mps()
