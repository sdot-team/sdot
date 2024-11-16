from ..bindings.loader import module_for
from .Distribution import Distribution

class ConstantValue( Distribution ):
    """
        `value` everywhere
    """

    def __init__( self, value = 1 ):
        self.value = value

    def binding( self, base_cell, binding_module ):
        m = module_for( "TF_objects", scalar_type = binding_module.dtype() )
        return m.ConstantValue( self.value )
