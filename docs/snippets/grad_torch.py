import torch
import sdot

positions = torch.rand( 100, 2, dtype = torch.float64, requires_grad = True )
masses    = torch.ones( 100, dtype = torch.float64, requires_grad = True )   # normalized by sdot: only the ratios count
pixels    = torch.rand( 10, 10, dtype = torch.float64, requires_grad = True )

value = sdot.w2( sdot.SumOfDiracs( positions, masses ), sdot.Image( pixels ) )
value.backward()

positions.grad, masses.grad, pixels.grad           # [ 100, 2 ], [ 100 ], [ 10, 10 ]
