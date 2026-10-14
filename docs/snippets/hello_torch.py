import torch
import sdot

f = sdot.SumOfDiracs( torch.rand( 100, 2 ) )  # 100 diracs of equal mass
g = sdot.Image( torch.rand( 10, 10 ) )  # a 10 x 10 image, on the unit square

print( sdot.w2( f, g ) )  # W2^2, as a torch tensor
