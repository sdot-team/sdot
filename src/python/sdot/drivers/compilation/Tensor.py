from ...Dyn import Dyn

class Tensor:
    """Sentinel type for returning a array from driver.call.

    Usage:
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    """

    @staticmethod
    def configure_call_ret_for( call_arg, fai, driver, shape, dtype = None ):
        axis_names = []
        for size in shape:
            if isinstance( size, Dyn ):
                axis_names.append( size.name )
            else:
                axis_names.append( "" )

        call_arg.configure_as_output_tensor( fai, driver, shape, dtype, axis_names, list_of_dynamic_axes = fai.global_list_of_dynamic_axes )
