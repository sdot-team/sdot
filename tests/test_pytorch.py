# import pytest
import torch
import sys
import os

# Add build and python/pytorch to path
sys.path.append( os.path.join( os.getcwd(), 'python/pytorch' ) )
sys.path.append( os.path.join( os.getcwd(), 'build' ) )

from sdot_pytorch import sdot_w2

def test_sdot_w2_cpu():
    dirac_xs = torch.tensor( [ 0.0 ], requires_grad = True )
    dirac_ws = torch.tensor( [ 1.0 ] )
    point_xs = torch.tensor( [ 0.0, 1.0 ] )
    point_ys = torch.tensor( [ 1.0, 1.0 ] )

    loss = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys )
    assert abs( loss.item() - 1 / 3 ) < 1e-6

    loss.backward()
    assert abs( dirac_xs.grad.item() + 1 ) < 1e-6

def test_sdot_w2_cpu_grad_check():
    def test( dirac_xs, dirac_ws, point_xs, point_ys ):
        torch.autograd.gradcheck( sdot_w2, (
            torch.tensor( dirac_xs, requires_grad = True ),
            torch.tensor( dirac_ws ),
            torch.tensor( point_xs ),
            torch.tensor( point_ys )
        ), eps = 1e-05, atol = 1e-05, rtol = 1e-3 )

    test( [ 0.0 ], [ 1.0 ], [ 0.0, 1.0 ],  [ 1.0, 1.0 ] )
    test( [ 0.2 ], [ 1.0 ], [ 0.0, 1.0 ],  [ 1.0, 1.0 ] )

