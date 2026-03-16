# import sdot


# f = sdot.PiecewiseAffineFunction( [ 0, 1 ], [ 1, 1 ] )
# g = sdot.SumOfWeightedDiracs( [ [ 0 ], [ 0 ] ] )
# from sdot_pytorch import sdot_w2
# import torch

# def test_w2_pytorch_1_dirac_1_1():
#     dirac_xs = torch.tensor( [ [ 0 ], [ 1 / 2 ] ], dtype = torch.float, requires_grad = True )
#     dirac_ws = torch.tensor( [ [ 1 ], [ 1 ] ], dtype = torch.float, requires_grad = True )
#     point_xs = torch.tensor( [ [ 0, 1 ], [ 0, 1 ] ], dtype = torch.float )
#     point_ys = torch.tensor( [ [ 1, 1 ], [ 1, 1 ] ], dtype = torch.float )

#     # forward
#     dist, bary = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=True )
#     assert torch.allclose( dist, torch.tensor( [ 1 / 3, 1 / 12 ] ) )
#     assert torch.allclose( bary, torch.tensor( [ 1 / 2, 1 / 2 ] ) )

#     # backward
#     torch.sum( dist ).backward()
#     assert torch.allclose( dirac_xs.grad, torch.tensor( [ [ - 1.0 ], [ 0.0 ] ] ) )

# def test_w2_pytorch_1_dirac_0_1():
#     dirac_xs = torch.tensor( [ [ 0 ], [ 1 / 2 ] ], dtype = torch.float, requires_grad = True )
#     dirac_ws = torch.tensor( [ [ 1 ], [ 1 ] ], dtype = torch.float, requires_grad = True )
#     point_xs = torch.tensor( [ [ 0, 1 ], [ 0, 1 ] ], dtype = torch.float )
#     point_ys = torch.tensor( [ [ 0, 2 ], [ 0, 2 ] ], dtype = torch.float )

#     # forward
#     dist, bary = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=True )
#     assert torch.allclose( dist, torch.tensor( [ 1 / 2, 1 / 12 ] ) )
#     assert torch.allclose( bary, torch.tensor( [ 2 / 3, 2 / 3 ] ) )

# def test_w2_pytorch_10_dirac_1_1():
#     dirac_xs = torch.linspace( 0.05, 0.95, 10, dtype = torch.float, requires_grad = True )
#     dirac_ws = torch.ones_like( dirac_xs, dtype = torch.float )
#     point_xs = torch.tensor( [ 0, 1 ], dtype = torch.float )
#     point_ys = torch.tensor( [ 1, 1 ], dtype = torch.float )

#     # forward
#     dist, bary = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=True )
#     assert torch.allclose( dist, torch.tensor( [ 1 / 1200 ] ) )
#     assert torch.allclose( bary, torch.linspace( 0.05, 0.95, 10 ) )

# def test_w2_pytorch_10_dirac_0_1():
#     dirac_xs = torch.linspace( 0.05, 0.95, 10, dtype = torch.float, requires_grad = True )
#     dirac_ws = torch.ones_like( dirac_xs, dtype = torch.float )
#     point_xs = torch.tensor( [ 0, 1 ], dtype = torch.float )
#     point_ys = torch.tensor( [ 0, 2 ], dtype = torch.float )

#     # forward
#     dist = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys )
#     assert torch.allclose( dist, torch.tensor( [ 0.03405928239226341248 ] ) )

# def test_w2_pytorch_10_dirac_0_1_d100():
#     p = torch.linspace( 0, 1, 11, dtype = torch.float )
#     dirac_xs = torch.linspace( 0.05, 0.95, 10, dtype = torch.float, requires_grad = True )
#     dirac_ws = torch.ones_like( dirac_xs, dtype = torch.float )
#     point_xs = p.clone()
#     point_ys = 2 * p

#     # forward
#     dist = sdot_w2( dirac_xs, dirac_ws, point_xs, point_ys )
#     # torch.set_printoptions( precision=20 )
#     assert torch.allclose( dist, torch.tensor( [ 0.03405928239226341248 ] ) )
