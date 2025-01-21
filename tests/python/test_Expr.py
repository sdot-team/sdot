from sdot.bindings import loader
from sdot import Expr
import pytest
import numpy

loader.set_auto_rebuild( True )

print( Expr( 10 ) )
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
