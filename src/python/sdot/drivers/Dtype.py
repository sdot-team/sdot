
from numpy import sign


class Dtype:
    """
    """

    def __init__( self, floating_point: bool, size: int, signed = True, driver_version = None ) -> None:
        self.driver_version = driver_version # will be update during driver instantiation
        self.floating_point = floating_point
        self.signed = signed
        self.size = size

    @staticmethod
    def factory( value ) -> 'Dtype':
        if isinstance( value, Dtype ):
            return Dtype( value.floating_point, value.size, value.signed, value.driver_version )

        if value is float:
            from .driver import driver
            return driver.ftype

        if value is int:
            from .driver import driver
            return driver.itype

        # -------------- str --------------
        value = str( value ).lower()

        if value == "int":
            return Dtype.si( size = 64 )

        if value == "float":
            return Dtype.fp( size = 32 )

        if value == "double":
            return Dtype.fp( size = 64 )

        if value.startswith( "fp" ):
            return Dtype.fp( size = int( value[ 2: ] ) )

        if value.startswith( "pi" ):
            return Dtype.pi( size = int( value[ 2: ] ) )

        if value.startswith( "si" ):
            return Dtype.si( size = int( value[ 2: ] ) )

        raise ValueError( f"unsupported type name: { str( value ) }" )

    @staticmethod
    def fp( size: int ):
        """ make a floating point type """
        return Dtype( floating_point = True, size = size )

    @staticmethod
    def si( size: int ):
        """ make a signed integer type """
        return Dtype( floating_point = False, size = size, signed = True )

    @staticmethod
    def pi( size: int ):
        """ make a signed integer type """
        return Dtype( floating_point = False, size = size, signed = False )

    @property
    def name( self ):
        return self.cpp_name

    @property
    def signature( self ):
        return self.cpp_name

    @property
    def cpp_name( self ):
        if self.floating_point:
            return f"FP{ self.size }"
        return f"SI{ self.size }"

    def __eq__( self, value, / ) -> bool:
        if not isinstance( value, Dtype ):
            value = Dtype.factory( value )
        assert isinstance( value, Dtype )
        return self.floating_point == value.floating_point and self.size == value.size

    def __neq__( self, value, / ) -> bool:
        return not self.__eq__( value )

    def jax_ffi_tensor_type( self ) -> str:
        if self.floating_point:
            return f"xla::ffi::F{ self.size }"
        if self.signed:
            return f"xla::ffi::S{ self.size }"
        return f"xla::ffi::U{ self.size }"
