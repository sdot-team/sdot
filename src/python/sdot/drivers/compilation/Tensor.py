
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
        call_arg.configure_as_output_tensor( fai, driver, shape, dtype )
