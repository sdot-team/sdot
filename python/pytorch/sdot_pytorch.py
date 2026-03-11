import torch
from torch.autograd import Function
import sdot_pytorch_cpp

class SDOTW2Function(Function):
    @staticmethod
    def forward(ctx, Xf, Wf, Xg, Yg):
        dist, barycenters = sdot_pytorch_cpp.forward(Xf, Wf, Xg, Yg)
        ctx.save_for_backward(Xf, Wf, Xg, Yg, barycenters)
        return dist, barycenters

    @staticmethod
    def backward(ctx, grad_dist, grad_bary):
        Xf, Wf, Xg, Yg, barycenters = ctx.saved_tensors
        grad_Xf, grad_Wf, grad_Xg, grad_Yg = sdot_pytorch_cpp.backward(
            grad_dist, grad_bary, Xf, Wf, Xg, Yg, barycenters)
        return grad_Xf, grad_Wf, grad_Xg, grad_Yg

def sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=False):
    dist, bary = SDOTW2Function.apply(Xf, Wf, Xg, Yg)
    if return_barycenters:
        return dist, bary
    return dist
