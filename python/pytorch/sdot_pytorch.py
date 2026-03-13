from torch.autograd import Function
import sdot_pytorch_bindings
import torch

class SDOTW2Function(Function):
    @staticmethod
    def forward(ctx, dirac_xs, dirac_ws, point_xs, point_ys):
        # Create output tensors
        dist = torch.empty( dirac_xs.shape[ :-1 ], dtype = dirac_xs.dtype, device = dirac_xs.device )
        bary = torch.empty_like( dirac_xs )

        # Call C++ implementation
        sdot_pytorch_bindings.forward_impl(
            dirac_xs.contiguous(), dirac_ws.contiguous(),
            point_xs.contiguous(), point_ys.contiguous(),
            dist, bary
        )

        ctx.save_for_backward(bary, dirac_xs, dirac_ws, point_xs, point_ys)
        return dist, bary

    @staticmethod
    def backward(ctx, grad_distance, grad_barycenters):
        barycenters, dirac_xs, dirac_ws, point_xs, point_ys = ctx.saved_tensors

        # Create output gradient tensors
        grad_dirac_xs = torch.empty_like(dirac_xs)
        grad_dirac_ws = torch.empty_like(dirac_ws)
        grad_point_xs = torch.empty_like(point_xs)
        grad_point_ys = torch.empty_like(point_ys)

        # Call C++ backward implementation
        sdot_pytorch_bindings.backward_impl(
            grad_distance.contiguous(), grad_barycenters.contiguous(),
            barycenters.contiguous(),
            dirac_xs.contiguous(), dirac_ws.contiguous(),
            point_xs.contiguous(), point_ys.contiguous(),
            grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys
        )

        return grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys

def sdot_w2(dirac_xs, dirac_ws, point_xs, point_ys, return_barycenters = False ):
    dist, bary = SDOTW2Function.apply(dirac_xs, dirac_ws, point_xs, point_ys)
    if return_barycenters:
        return dist, bary
    return dist
