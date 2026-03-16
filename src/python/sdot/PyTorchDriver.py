import torch

class PyTorchDriver:
    """

    """
    def __init__( self, dtype = torch.float32, device = None ):
        self.device = device or torch.get_default_device()
        self.dtype = dtype

    def t2( self, tensor ):
        """ make a rank 2 + 1 (for the batches) tensor """
        if tensor is None:
            return tensor
        res = torch.as_tensor( tensor, dtype = self.dtype, device = self.device )
        if res.ndim == 2:
            res = res[ None, :, : ]
        assert( res.ndim == 3 )
        return res

    def t1( self, tensor ):
        """ make a rank 1 + 1 (for the batches) tensor """
        if tensor is None:
            return tensor
        res = torch.as_tensor( tensor, dtype = self.dtype, device = self.device )
        if res.ndim == 1:
            res = res[ None, : ]
        assert( res.ndim == 2 )
        return res

    def ones( self, shape ):
        return torch.ones( shape, dtype = self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return torch.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape ):
        return torch.ones( shape, dtype = self.dtype, device = self.device )
