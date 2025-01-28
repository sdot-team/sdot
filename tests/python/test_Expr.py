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

