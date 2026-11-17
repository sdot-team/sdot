"""The radial functions of pysdot ( `pysdot.radial_funcs` ): what each cell integrates, as a function of the distance to its seed.

Three of them have an equivalent in sdot, and are what `PowerDiagram` and `OptimalTransport` accept:

- `RadialFuncUnit`: the density alone ( the default );
- `RadialFuncInBall`: the density inside the ball `|x - p_i|^2 <= w_i` only -- a partial transport, `sink_weight = 0` in sdot;
- `RadialFuncEntropy( eps )`: the density times `exp( ( w_i - |x - p_i|^2 ) / eps )` -- `sdot.RadialExp( eps / 2 )`.

The others are kept so that importing them does not fail, but a diagram that uses them raises `NotImplementedError`.
"""


class RadialFuncUnit:
    def name( self ):
        return "1"

    def need_rb_corr( self ):
        return True

    def second_order_moment_name( self ):
        return "r^2"

    def ball_cut( self ):
        return False


class RadialFuncInBall:
    def name( self ):
        return "in_ball(weight**0.5)"

    def need_rb_corr( self ):
        return False

    def second_order_moment_name( self ):
        return "r^2*in_ball(weight**0.5)"

    def ball_cut( self ):
        return True


class RadialFuncEntropy:
    def __init__( self, eps ):
        self.eps = float( eps )
        self.s = "exp((w-r**2)/{})".format( eps )

    def name( self ):
        return self.s

    def need_rb_corr( self ):
        return False

    def ball_cut( self ):
        return False


class _Unsupported:
    """a radial function of pysdot that sdot does not integrate ( yet )"""
    def need_rb_corr( self ):
        return False

    def ball_cut( self ):
        return False

    def second_order_moment_name( self ):
        return ""


class RadialFuncPpWmR2( _Unsupported ):
    def name( self ):
        return "pos_part(w-r**2)"

    def ball_cut( self ):
        return True


class RadialFuncR2( _Unsupported ):
    def __init__( self, eps = None ):
        pass

    def name( self ):
        return "r**2"


class RadialFuncArfd( _Unsupported ):
    def __init__( self, *args, **kwargs ):
        pass

    def name( self ):
        return "arfd"


class CompressibleFunc( _Unsupported ):
    def __init__( self, kappa, gamma, g, f_cor, pi_0 = 1, c_p = 1 ):
        self.s = "compressible_func({:.16f} {:.16f} {:.16f} {:.16f} {:.16f} {:.16f})".format( kappa, gamma, g, f_cor, pi_0, c_p )

    def name( self ):
        return self.s


def sdot_radial_args( radial_func ):
    """`( sink_weight, radial_func )` of an sdot diagram for a pysdot radial function"""
    if radial_func is None or isinstance( radial_func, RadialFuncUnit ):
        return None, None
    if isinstance( radial_func, RadialFuncInBall ):
        return 0.0, None
    if isinstance( radial_func, RadialFuncEntropy ):
        from .. import RadialExp
        # pysdot: exp( ( w - r^2 ) / eps ); sdot: exp( ( w - r^2 ) / ( 2 eps' ) )
        return None, RadialExp( radial_func.eps / 2 )
    raise NotImplementedError( f"sdot.pysdot: { type( radial_func ).__name__ } has no equivalent in sdot "
                               "( RadialFuncUnit, RadialFuncInBall and RadialFuncEntropy do )" )
