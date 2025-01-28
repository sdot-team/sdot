from .bindings.loader import module_for, normalized_dtype, type_promote
from .TransformationMatrix import TransformationMatrix
# from types import ModuleType
import numpy as np

axis_symbols = []

class Expr:
    """ wrapper around the cpp class sdot::Expr used to store symbolic expressions """

    def __init__( self, value = None ):
        """ 
        """
        # default value
        if value is None:
            value = 0

        # already an Expr ?
        if isinstance( value, Expr ):
            self._expr = value._expr
            return

        _module = module_for( 'generic_objects' )
        if isinstance( value, _module.Expr ):
            self._expr = value
            return

        # else, call the cpp ctor
        self._expr = _module.Expr( value )

    def subs( self, symbol_map ):
        m = [ ( Expr( k )._expr, Expr( v )._expr ) for k, v in symbol_map ]
        return Expr( self._expr.subs( m ) )

    def constant_value( self ):
        """ if constant, return the value. Else, return None. """
        valid, value = self._expr.constant_value()
        if valid:
            return value
        return None

    def always_equal( self, that ):
        if not isinstance( that, Expr ):
            return self.always_equal( Expr( that ) )
        return self._expr.always_equal( that._expr )

    @property
    def natural_args( self ):
        l = self._expr.natural_args()
        if len( l ) != 1:
            return None
        return [ ( name, Expr( value ) ) for name, value in l[ 0 ] ]

    def set_natural_args( self, nargs ):
        """  """
        args = []
        for arg in nargs:
            if isinstance( arg, tuple ):
                args.append( ( str( arg[ 0 ] ), Expr( arg[ 1 ] )._expr ) )
            else:
                args.append( ( '', Expr( arg )._expr ) )

        self._expr.set_natural_args( args )

    @staticmethod
    def always_equal( a, b ):
        if not isinstance( a, Expr ):
            a = Expr( a )
        if not isinstance( b, Expr ):
            b = Expr( b )
        return Expr( a._expr.always_equal( b._expr ) )

    @staticmethod
    def alternative( index, *expr_list ):
        """ generalisation of the `?:` C++ operator. """
        return Expr( Expr( index )._expr.alternative( [ Expr( x )._expr for x in expr_list ] ) )

    @staticmethod
    def ceil( value ):
        """ ceil unary operator. """
        return Expr( Expr( value )._expr.ceil() )

    @staticmethod
    def frac( value ):
        """ fractional unary operator (ex: 1.7 -> 0.7). """
        return Expr( Expr( value )._expr.frac() )

    @staticmethod
    def logical_and( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.and_boolean( that._expr ) )

    @staticmethod
    def logical_or( a, b ):
        if not isinstance( a, Expr ):
            a = Expr( a )
        if not isinstance( b, Expr ):
            b = Expr( b )
        return Expr( a._expr.or_boolean( b._expr ) )

    @staticmethod
    def axis( n ):
        """ get the default symbol for axis `n`
        """
        while len( axis_symbols ) <= n:
            axis_symbols.append( Expr( f'x_{ len( axis_symbols ) }' ) )
        return axis_symbols[ n ]

    @staticmethod
    def array( array, indices = None, exterior = "zero value", interpolation = None, overriding_formula = None ):
        """ symbolic expression from an array

            `indices` must be a list of expressions. If not specified, it will be equal to [ Expr.axis( 0 ), Expr.axis( 1 ), ... ]. 

            `overriding_formula` can be used to say explicitely what to do with the indices
                def ovf( a, x ): # a is a raw symbolic array, with no  
                    i = Expr.ceil( x )
                    f = Expr.frac( x )
                    return a[ i + 0 ] * ( 1 - f ) + a[ i + 1 ] * f

                v = Expr.array( [ 10, 20, 30 ], overriding_formula = ovf )
                print( v )


            Currently, exterior can be either
            * "zero value"
            * "unchecked"
            * "projected"
            * "periodic"

            Interpolation can be either
            * None or "P0" for piecewise constant interpolation
            * "P1" for piecewise affine interpolation
        """
        # array type
        array = np.ascontiguousarray( array )

        # need to load the module to get the Expr type
        module_for( 'generic_objects' )

        # make a base array with local indices
        module = module_for( 'symbolic_array', 
            scalar_type = type_promote( [ array.dtype ] ), 
            nb_dims = array.ndim
        )

        base_indices = [ Expr( f"bi_{ i }" ) for i in range( array.ndim ) ]
        base_array = Expr( module.symbolic_array( array, [ b._expr for b in base_indices ] ) )

        # indices
        if indices is None:
            indices = [ Expr.axis( i ) for i in range( array.ndim ) ]

        # if overriding_formula is specified, use it
        if overriding_formula is not None:
            assert interpolation is None
            assert exterior is None

            ovf = overriding_formula( base_array, *base_indices )
            ovf = ovf.subs( zip( base_indices, indices ) )
            if ovf.natural_args is None:
                ovf.set_natural_args( indices )
            return ovf

        # exterior values
        if exterior == "zero value":
            cond = Expr( 1 )
            for i in range( array.ndim ):
                cond = Expr.logical_and( cond, base_indices[ i ] < array.shape[ i ] )
                cond = Expr.logical_and( cond, base_indices[ i ] >= 0 )
            base_array = Expr.alternative( cond, 0, base_array )
        elif exterior == "projected":
            TODO()
        elif exterior == "periodic":
            TODO()
        elif exterior != "unchecked":
            raise ValueError( "Unknown exterior type" )

        # interpolation
        if interpolation is not None:
            if interpolation == "P1": # piecewise affine
                loc_inds = [ Expr( f"li_{ i }" ) for i in range( array.ndim ) ]
                r = Expr( 0 )
                for n in range( 2 ** array.ndim ):
                    l = [ Expr.ceil( loc_inds[ i ] - 0.5 ) + bool( n & 2**i ) for i in range( array.ndim ) ]
                    m = base_array.__getitem__( tuple( l ) )
                    for i in range( array.ndim ):
                        f = Expr.frac( loc_inds[ i ] - 0.5 )
                        if n & 2**i:
                            m *= f
                        else:
                            m *= 1 - f
                    r += m

                base_array = r.subs( zip( loc_inds, base_indices ) )
            elif interpolation != "P0":
                raise ValueError( "Unknown interpolation type" )

        # final indices
        res = base_array.subs( zip( base_indices, indices ) )
        if res.natural_args is None:
            res.set_natural_args( indices )
        return res

    @staticmethod
    def list_from_compact_repr( crepr ):
        _module = module_for( 'generic_objects' )
        return _module.expr_list_from_compact_repr( crepr )
    
    @staticmethod
    def as_expr( expr ):
        if expr is None:
            return Expr( "0" )
        if isinstance( expr, Expr ):
            return expr
        if isinstance( expr, list ):
            return [ Expr.as_expr( v ) for v in expr ]
        return Expr( expr )

    @staticmethod
    def ct_rt_split_of_list( expr_list ):
        module = module_for( 'generic_objects' )
        return module.ct_rt_split_of_list( [ Expr.as_expr( e )._expr for e in expr_list ] )

    @staticmethod
    def cell_splits_of_list( funcs, rt_data ):
        """ splits, final_funcs """
        module = module_for( 'generic_objects' )
        return module.cell_splits_of_list( [ Expr.as_expr( e )._expr for e in funcs ], rt_data )

    def ct_repr( self ):
        return self._expr.ct_repr()

    def rt_data( self ):
        return self._expr.rt_data()

    def boundary_split( self, ndim ):
        return self._expr.boundary_split( ndim )

    def __repr__( self ):
        return self._expr.__repr__()

    def __add__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.add( that._expr ) )

    def __radd__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( that._expr.add( self._expr ) )

    def __sub__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.sub( that._expr ) )

    def __rsub__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( that._expr.sub( self._expr ) )

    def __neg__( self ):
        return Expr( self._expr.neg() )

    def __mul__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.mul( that._expr ) )

    def __rmul__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.mul( that._expr ) )

    def __truediv__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.div( that._expr ) )
    
    def __pow__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.pow( that._expr ) )
    
    def __equal__( self, that ):
        if not isinstance( that, Expr ):
            that = Expr( that )
        return Expr( self._expr.equal( that._expr ) )

    def __getitem__( self, ind ):
        """ substitute "natural entries" of self with index / indices """
        m = []
        if isinstance( ind, tuple ):
            for arg in ind:
                m.append( ( '', Expr( arg )._expr ) )
        else:
            m.append( ( '', Expr( ind )._expr ) )
        return Expr( self._expr.apply( m ) )

    def __call__( self, *largs, **margs ):
        """ substitute "natural entries" of self with values in largs and margs """
        m = []
        for arg in largs:
            m.append( ( '', Expr( arg )._expr ) )
        for name, arg in margs.items():
            m.append( ( name, Expr( arg )._expr ) )
        return Expr( self._expr.apply( m ) )

    def __eq__( self, other ):
        if not isinstance( other, Expr ):
            return self.__eq__( Expr( other ) )
        return self._expr.equal( other._expr )

    def __ge__( self, other ):
        if not isinstance( other, Expr ):
            return self.__ge__( Expr( other ) )
        return self._expr.supeq( other._expr )

    def __le__( self, other ):
        if not isinstance( other, Expr ):
            return self.__le__( Expr( other ) )
        return self._expr.infeq( other._expr )

    def __ne__( self, other ):
        if not isinstance( other, Expr ):
            return self.__ne__( Expr( other ) )
        return self._expr.neq( other._expr )

    def __gt__( self, other ):
        if not isinstance( other, Expr ):
            return self.__gt__( Expr( other ) )
        return self._expr.sup( other._expr )

    def __lt__( self, other ):
        if not isinstance( other, Expr ):
            return self.__lt__( Expr( other ) )
        return self._expr.inf( other._expr )
