from __future__ import annotations
from dataclasses import dataclass
from .CtKnown import CtKnown
import ast

class ShapeItem:
    """ handles axis names like
        3 * dim + nb_elements[ index < dim, _ < nb_elements ] + 1 + nb_items[ index < dim ]
    """

    @dataclass
    class Term:
        selection: list[ ShapeItem ] | None # index constraints, e.g. [index < dim, _ < nb_elements]
        coeff: int = 1
        name: str = ''

    ct_known_limit: int
    ct_known: bool
    offset: int # constant offset
    terms: list[ ShapeItem.Term ] = []
    index: ShapeItem | None # e.g. "index" in "index < dim"

    def __init__( self, value ):
        self.ct_known_limit = -1
        self.ct_known = False
        self.index = None
        self.terms = []
        self.offset = 0

        if isinstance( value, CtKnown ):
            self.ct_known_limit = value.limit
            self.ct_known = True
            value = value.value

        if isinstance( value, int ):
            self.offset = int( value )
            return

        if isinstance( value, str ):
            self._parse( ast.parse( value, mode='eval' ).body )

    def ndim( self ):
        return 1

    def get_axis_names( self, axis_names: set[ str ] ):
        for term in self.terms:
            name = term.name
            if term.selection is not None:
                name = "max_of_" + name

            axis_names.add( name )

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        for term in self.terms:
            axes[ term.name ] = term.selection
        if self.ct_known:
            for term in self.terms:
                ct_axes[ term.name ] = self.ct_known_limit

    def _parse( self, node ):
        match node:
            case ast.BinOp( op=ast.Add(), left=l, right=r ):
                self._parse( l )
                self._parse( r )
            case ast.BinOp( op=ast.Mult(), left=ast.Constant( value=int( n ) ), right=rest ):
                self.terms.append( ShapeItem._term_from_node( rest, n ) )
            case ast.BinOp( op=ast.Mult(), left=rest, right=ast.Constant( value=int( n ) ) ):
                self.terms.append( ShapeItem._term_from_node( rest, n ) )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                self.terms.append( ShapeItem._term_from_subscript( name, cmp ) )
            case ast.Name( id=name ):
                self.terms.append( ShapeItem.Term( None, 1, name ) )
            case ast.Constant( value=int( n ) ):
                self.offset += n
            case ast.Compare( left=left_expr, ops=[ ast.Lt() ], comparators=[ right_expr ] ):
                self.index = ShapeItem._from_node( right_expr )
                self._parse( left_expr )
            case _:
                raise ValueError( f"unsupported expression: {ast.dump( node )}" )

    @staticmethod
    def _term_from_node( node, coeff: int = 1 ) -> 'ShapeItem.Term':
        match node:
            case ast.Name( id=name ):
                return ShapeItem.Term( None, coeff, name )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                return ShapeItem._term_from_subscript( name, cmp, coeff )
            case _:
                raise ValueError( f"unsupported: {ast.dump( node )}" )

    @staticmethod
    def _term_from_subscript( name: str, cmp, coeff: int = 1 ) -> 'ShapeItem.Term':
        if isinstance( cmp, ast.Tuple ):
            return ShapeItem.Term( [ ShapeItem._from_node( elt ) for elt in cmp.elts ], coeff, name )
        return ShapeItem.Term( [ ShapeItem._from_node( cmp ) ], coeff, name )

    @staticmethod
    def _from_node( node ) -> 'ShapeItem':
        s = object.__new__( ShapeItem )
        s.offset = 0
        s.terms = []
        s.index = None
        s._parse( node )
        return s
