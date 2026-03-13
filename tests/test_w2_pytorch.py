from sdot_pytorch import sdot_w2
import torch

def test_w2_pytorch():
    dirac_xs = torch.tensor( [ [ 0 ], [ 1 / 2 ] ], dtype = torch.float, requires_grad = True )
    dirac_ws = torch.tensor( [ [ 1 ], [ 1 ] ], dtype = torch.float, requires_grad = True )
    point_xs = torch.tensor( [ [ 0, 1 ], [ 0, 1 ] ], dtype = torch.float )
    point_ys = torch.tensor( [ [ 1, 1 ], [ 1, 1 ] ], dtype = torch.float )

    # Test distance
    dist = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys )
    assert torch.allclose( dist, torch.tensor( [ 1 / 3, 1 / 12 ] ) )

    # Test barycenters
    dist, bary = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=True )
    assert torch.allclose( bary, torch.tensor( [ 1 / 2, 1 / 2 ] ) )

    # gradiens
    torch.sum( dist ).backward()
    assert torch.allclose( dirac_xs.grad, torch.tensor( [ [ - 1.0 ], [ 0.0 ] ] ) )
