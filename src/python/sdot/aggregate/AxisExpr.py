from __future__ import annotations

from .AxisVariable import AxisVariable
from dataclasses import dataclass
import ast

class AxisExpr:
    """
    An expression defining the size of one or more axes, possibly depending on axis variables.

    Example: `3 * dim + nb_elements[ dim, 2 * nb_points ] + 1 + nb_items`

    Notation:
        `[ ... ]`  Dynamic value — the variable becomes a tensor whose rank equals the
                   number of index arguments; a `max_of_<name>` accessor is also generated.

        `( ... )`  Expansion — the variable is replicated into one axis per value.
                   `nb_knots( dim ) + 1`  →  `nb_knots_0 + 1`, ..., `nb_knots_{dim-1} + 1`
    """

    @dataclass
    class Term:
        variable: AxisVariable
        coeff: int

    offset: int
    terms: list[ AxisExpr.Term ]

    def __init__( self, value ):
        if isinstance( value, AxisExpr ):
            self.offset = value.offset
            self.terms = value.terms
            return

        self.offset = 0
        self.terms = []

        if isinstance( value, int ):
            self.offset = int( value )
            return

        if isinstance( value, str ):
            self._parse( ast.parse( value, mode='eval' ).body )

    def ndim( self ) -> int:
        return 1

    def get_axis_names( self, axis_names: set[ str ] ):
        for term in self.terms:
            name = term.variable.name
            if term.variable.selection is not None:
                name = "max_of_" + name
            axis_names.add( name )

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        for term in self.terms:
            axes[ term.variable.name ] = term.variable.selection

    def _parse( self, node ):
        match node:
            case ast.BinOp( op=ast.Add(), left=l, right=r ):
                self._parse( l )
                self._parse( r )
            case ast.BinOp( op=ast.Mult(), left=ast.Constant( value=int( n ) ), right=rest ):
                self.terms.append( AxisExpr._term_from_node( rest, n ) )
            case ast.BinOp( op=ast.Mult(), left=rest, right=ast.Constant( value=int( n ) ) ):
                self.terms.append( AxisExpr._term_from_node( rest, n ) )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                self.terms.append( AxisExpr._term_from_subscript( name, cmp ) )
            case ast.Call( func=ast.Name( id=name ), args=args, keywords=[] ):
                self.terms.append( AxisExpr._term_from_call( name, args ) )
            case ast.Name( id=name ):
                self.terms.append( AxisExpr.Term( AxisVariable( None, None, name ), 1 ) )
            case ast.Constant( value=int( n ) ):
                self.offset += n
            case _:
                raise ValueError( f"unsupported expression: {ast.dump( node )}" )

    @staticmethod
    def _term_from_node( node, coeff: int = 1 ) -> 'AxisExpr.Term':
        match node:
            case ast.Name( id=name ):
                return AxisExpr.Term( AxisVariable( None, None, name ), coeff )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                return AxisExpr._term_from_subscript( name, cmp, coeff )
            case ast.Call( func=ast.Name( id=name ), args=args, keywords=[] ):
                return AxisExpr._term_from_call( name, args, coeff )
            case _:
                raise ValueError( f"unsupported: {ast.dump( node )}" )

    @staticmethod
    def _term_from_subscript( name: str, cmp, coeff: int = 1 ) -> 'AxisExpr.Term':
        if isinstance( cmp, ast.Tuple ):
            selection = [ AxisExpr._from_node( elt ) for elt in cmp.elts ]
        else:
            selection = [ AxisExpr._from_node( cmp ) ]
        return AxisExpr.Term( AxisVariable( None, selection, name ), coeff )

    @staticmethod
    def _term_from_call( name: str, args, coeff: int = 1 ) -> 'AxisExpr.Term':
        arguments = [ AxisExpr._from_node( arg ) for arg in args ]
        return AxisExpr.Term( AxisVariable( arguments, None, name ), coeff )

    @staticmethod
    def _from_node( node ) -> 'AxisExpr':
        s = object.__new__( AxisExpr )
        s.offset = 0
        s.terms = []
        s._parse( node )
        return s
