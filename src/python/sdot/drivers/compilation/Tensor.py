# from ...CtKnown import CtKnown
from ...Dyn import Dyn

class Tensor:
    """

    Usage:
        driver.call( "measure", includes,
            Return( Tensor( shape=[], dtype=float ) ),
            cell
        )
    """

    @staticmethod
    def configure_call_ret_for( call_arg, fai, driver, shape, dtype = None, axis_names = None, represents_a_dynamic_axis = False ):
        ct_axes = {}
        if axis_names is None:
            axis_names = []
            for size in shape:
                if isinstance( size, Dyn ):
                    axis_names.append( size.name + "_capacity" )
                else:
                    axis_names.append( "" )

        call_arg.configure_as_output_tensor( fai, driver, shape, dtype, axis_names = axis_names, ct_axes = ct_axes, represents_a_dynamic_axis = represents_a_dynamic_axis )
