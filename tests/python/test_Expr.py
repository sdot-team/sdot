from sdot.bindings import loader
from sdot import Expr
import pytest
import numpy

loader.set_auto_rebuild( True )

def test_simplications():
    a = Expr( "a" )
    b = Expr( "b" )
    assert str( a - 20 + b + a ) == str( 2 * a + b - 20 )
    assert str( a - 20 - a ) == str( - 20 )

a = Expr( "a" )
b = Expr( "b" )
c = Expr( "c" )

# print( Expr.alternative( c, a, b ) )
# print( Expr.alternative( c, a, b ).subs( [ ( c, 0 ) ] ) )
# print( Expr.alternative( c, a, b ).subs( [ ( c, 1 ) ] ) )

# # e = a + 20
# # f = a + 21
# # print( 21 * ( b + 1 ) * a * ( b + 1 ) )
# # print( a - a )
# # print( a == b )
# # print( a == a )
# # print( a > a )
# # print( Expr( 10 ) == 10 )
# # print( Expr.always_equal( e, f ) )

# def ovf( a, x ):
#     i = Expr.ceil( x )
#     f = Expr.frac( x )
#     return a[ i + 0 ] * ( 1 - f ) + a[ i + 1 ] * f

v = Expr.array( [ 10, 20, 30 ], interpolation = "P1" )
print( v )
print( v[ 0.0 ] )
print( v[ 0.5 ] )
print( v[ 1.0 ] )
# print( v.natural_args )
# print( v[ 0 ] )
# print( v[ 0.2 ] )
# print( v[ 1 ] )


# print( v.subs( [ ( Expr.axis( 0 ), 0 ), ( Expr.axis( 1 ), 0 ) ] ) )
# print( v.subs( [ ( Expr.axis( 0 ), 1 ), ( Expr.axis( 1 ), 0 ) ] ) )
# print( v.subs( [ ( Expr.axis( 0 ), 1 ), ( Expr.axis( 1 ), 1 ) ] ) )
# print( v.subs( [ ( Expr.axis( 0 ), 2 ), ( Expr.axis( 1 ), 1 ) ] ) )

# print( v[ 0 ] )
# print( v[ 10 ] )
# print( v[ 20 ] )
# print( v[ 30 ] )
# print( v[ "yo" ] )
# # e = Expr( "x_0" )
# # print( v[ e ] )
# # f = Expr( "10" )

# # print( e )
# # # print( e * f )
# # # print( e - f )
# # print( e / f )
# # print( e[ 2, 3 ] )
