from .Distribution import Distribution

class ConstantValue( Distribution ):
    """
        `value` everywhere
    """

    def __init__( self, value = 1 ):
        self.value = value

    def binding( self, base_cell, binding_module ):
        return binding_module.ConstantValue( self.value )
