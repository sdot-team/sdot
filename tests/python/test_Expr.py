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
# e = a + 20
# f = a + 21
# print( 21 * ( b + 1 ) * a * ( b + 1 ) )
print( a - a )
print( a == b )
print( a == a )
print( a > a )
print( Expr( 10 ) == 10 )
# print( Expr.always_equal( e, f ) )

# v = Expr.array( [ 1, 2 ], [ Expr( "x_0" ) / 10 ] )
# print( v )
# # print( v.subs( { "x_0": 0 } ) )
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
