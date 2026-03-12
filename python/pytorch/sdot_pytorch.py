import torch
from torch.autograd import Function
import sdot_pytorch_cpp

class SDOTW2Function(Function):
    @staticmethod
    def forward(ctx, dirac_xs, dirac_ws, point_xs, point_ys):
        distance, barycenters = sdot_pytorch_cpp.forward(dirac_xs, dirac_ws, point_xs, point_ys)
        ctx.save_for_backward(dirac_xs, dirac_ws, point_xs, point_ys, barycenters)
        return distance, barycenters

    @staticmethod
    def backward(ctx, grad_distance, grad_barycenters):
        # grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
        return sdot_pytorch_cpp.backward( grad_distance, grad_barycenters, *ctx.saved_tensors )

def sdot_w2(dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=False):
    dist, bary = SDOTW2Function.apply(dirac_xs, dirac_ws, point_xs, point_ys)
    if return_barycenters:
        return dist, bary
    return dist

class SDOTL2Function(Function):
    @staticmethod
    def forward(ctx, f, g):
        res = sdot_pytorch_cpp.l2_forward(f, g)
        ctx.save_for_backward(f, g)
        return res[0]

    @staticmethod
    def backward(ctx, grad_output):
        f, g = ctx.saved_tensors
        grads = sdot_pytorch_cpp.l2_backward(f, g)
        return grads[0] * grad_output, grads[1] * grad_output

def sdot_l2(f, g):
    return SDOTL2Function.apply(f, g)
