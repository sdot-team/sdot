
class Tensor:
    """Sentinel type for returning a raw array from driver.call.

    Usage:
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    """

    @staticmethod
    def cpp_class_name_for( shape, dtype = None ):
        if dtype is int:
            return "MI"
        if dtype is not None:
            raise NotImplementedError
        return "MF"

    @classmethod
    def output_specs( cls, drv, shape=(), dtype=float ):
        return [ ( '', list( shape ), dtype ) ]

    @classmethod
    def from_outputs( cls, arrays, shape=(), dtype=float ):
        return arrays[ 0 ]
