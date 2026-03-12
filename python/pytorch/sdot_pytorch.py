from torch.autograd import Function
import sdot_pytorch_cpp
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # mandatory

class SDOTW2Function(Function):
    @staticmethod
    def forward(ctx, dirac_xs, dirac_ws, point_xs, point_ys):
        distance, barycenters = sdot_pytorch_cpp.forward(dirac_xs, dirac_ws, point_xs, point_ys)
        ctx.save_for_backward(barycenters, dirac_xs, dirac_ws, point_xs, point_ys)
        return distance, barycenters

    @staticmethod
    def backward(ctx, grad_distance, grad_barycenters):
        grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys = sdot_pytorch_cpp.backward( grad_distance, grad_barycenters, *ctx.saved_tensors )
        return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

def sdot_w2(dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters=False):
    dist, bary = SDOTW2Function.apply(dirac_xs, dirac_ws, point_xs, point_ys)
    if return_barycenters:
        return dist, bary
    return dist

