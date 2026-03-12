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
