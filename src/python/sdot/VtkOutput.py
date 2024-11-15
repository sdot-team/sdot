from .bindings.loader import module_for
# import numpy as np

class VtkOutput:
    """ A simple class to generate .vtk files 
    
    """

    def __init__( self ):
        """ 
        """

        self._module = module_for( 'generic_objects' )
        self._inst = self._module.VtkOutput()


    def save( self, filename ):
        self._inst.save( filename )
