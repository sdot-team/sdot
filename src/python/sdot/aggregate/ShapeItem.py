from __future__ import annotations
from dataclasses import dataclass
import ast

class ShapeItem:
    """ handles axis names like
        3 * dim + shape[ index < dim ] + 1 + nb_items[ index < dim ]
    """

    @dataclass
    class Selection:
        index: ShapeItem | None # e.g. parsing for "index < dim"
        name: 'str'

    @dataclass
    class Addition:
        selection: ShapeItem.Selection
        coeff: int = 1

    index: ShapeItem | None # e.g. parsing of "index"
    additions: list[ ShapeItem.Addition ] = []
    offset: int

    def __init__( self, value ):
        self.index = None
        self.additions = []
        self.offset = 0

        if isinstance( value, int ):
            self.offset = int( value )
            return

        if isinstance( value, str ):
            self._parse( ast.parse( value, mode='eval' ).body )

    def _parse( self, node ):
        match node:
            case ast.BinOp( op=ast.Add(), left=l, right=r ):
                self._parse( l )
                self._parse( r )
            case ast.BinOp( op=ast.Mult(), left=ast.Constant( value=int( n ) ), right=rest ):
                self.additions.append( ShapeItem.Addition( ShapeItem._selection_from_node( rest ), n ) )
            case ast.BinOp( op=ast.Mult(), left=rest, right=ast.Constant( value=int( n ) ) ):
                self.additions.append( ShapeItem.Addition( ShapeItem._selection_from_node( rest ), n ) )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                self.additions.append( ShapeItem.Addition( ShapeItem._selection_from_subscript( name, cmp ) ) )
            case ast.Name( id=name ):
                self.additions.append( ShapeItem.Addition( ShapeItem.Selection( None, name ) ) )
            case ast.Constant( value=int( n ) ):
                self.offset += n
            case ast.Compare( left=left_expr, ops=[ ast.Lt() ], comparators=[ right_expr ] ):
                self.index = ShapeItem._from_node( right_expr )
                self._parse( left_expr )
            case _:
                raise ValueError( f"unsupported expression: {ast.dump( node )}" )

    @staticmethod
    def _selection_from_node( node ) -> 'ShapeItem.Selection':
        match node:
            case ast.Name( id=name ):
                return ShapeItem.Selection( None, name )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                return ShapeItem._selection_from_subscript( name, cmp )
            case _:
                raise ValueError( f"unsupported: {ast.dump( node )}" )

    @staticmethod
    def _selection_from_subscript( name: str, cmp ) -> 'ShapeItem.Selection':
        return ShapeItem.Selection( ShapeItem._from_node( cmp ), name )

    @staticmethod
    def _from_node( node ) -> 'ShapeItem':
        s = object.__new__( ShapeItem )
        s.offset = 0
        s.additions = []
        s.index = None
        s._parse( node )
        return s
