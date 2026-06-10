from __future__ import annotations
from typing_extensions import Optional

from ..util import append_if_unique

from .AxisTerm import AxisTerm

import numpy
import ast
import re

class AxisExpr:
    """
    An expression defining the size of one or more axes, possibly depending on axis variables.

    Example: `3 * dim + nb_elements[ dim, 2 * nb_points ] + 1 + nb_items`

    It is a linear combination `offset + sum( coeff * variable )`, each `coeff * variable`
    stored as an `AxisTerm`.

    Notation:
        `[ ... ]`  Dynamic value — the variable becomes a tensor whose rank equals the
                   number of index arguments; a `max_of_<name>` accessor is also generated.

        `( ... )`  Expansion — the variable is replicated into one axis per value.
                   `nb_knots( dim ) + 1`  →  `nb_knots_0 + 1`, ..., `nb_knots_{dim-1} + 1`
    """

    offset: int
    terms: list[ AxisTerm ]

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
            self._parse( ast.parse( re.sub( r'\[\s*\]', '[()]', value ), mode='eval' ).body )

    @property
    def always_one( self ):
        return self.offset == 1 and len( self.terms ) == 0

    def unidimensional_version( self ):
        res = AxisExpr( self.offset )
        for term in self.terms:
            if term.name == "dim":
                res.offset += term.coeff
                continue
            res.terms.append( term.unidimensional_version() )
        return res

    def ndim( self, value_of_axis_variable ) -> int:
        res = 1
        for term in self.terms:
            if term.arguments:
                for argument in term.arguments:
                    # res *= argument.value( value_of_axis_variable, False )
                    raise NotImplementedError
        return res

    def value( self, system, forbidden_names ) -> Optional[ int ]:
        res = self.offset
        for term in self.terms:
            value = term.variable_value( system, forbidden_names )
            if value is None:
                return None
            res += term.coeff * value
        return res

    def solve( self, name, actual_shape_list, actual_shape_offset, system, forbidden_names: list[ str ], care_about_argument = True ) -> tuple[ Optional[ int | numpy.ndarray ], Optional[ int ] ]:
        """Return ( value, nb_terms_in_actual_shape_list )"""

        # argument ?
        if care_about_argument:
            argument = self.argument
            if argument is not None:
                length = argument.value( system, forbidden_names )
                # an expansion spans a statically known number of axes (e.g. `dim`); a symbolic
                # length (CppScalar) means we can't enumerate the axes -> leave it to the caller.
                if isinstance( length, int ):
                    res = numpy.empty( [ length ], dtype = int )
                    for index in range( length ):
                        value, _ = self.solve( name, actual_shape_list, actual_shape_offset + index, system, forbidden_names, care_about_argument = False )
                        if value is None:
                            return None, length
                        assert isinstance( value, int )
                        res[ index ] = value
                    return res, length

        # if we have `name` in some of the term...
        for num_term, term in enumerate( self.terms ):
            if term.exposed_name == name:
                lhs = actual_shape_list[ actual_shape_offset ]
                res = lhs - self.offset
                # ...and we can find the values for the other terms
                for num_mret, mret in enumerate( self.terms ):
                    if num_mret == num_term:
                        continue
                    v = mret.variable_value( system, forbidden_names + [ name ] )
                    if v is None:
                        return None, 1
                    res -= mret.coeff * v
                if isinstance( res, int ):
                    assert res % term.coeff == 0
                return res // term.coeff, 1
        return None, 1

    def get_axis_variable_names( self, axis_variable_names: list[ str ] ):
        for term in self.terms:
            append_if_unique( axis_variable_names, term.exposed_name )
            if term.arguments is not None:
                for argument in term.arguments:
                    argument.get_axis_variable_names( axis_variable_names )
            if term.selection is not None:
                for selection in term.selection:
                    selection.get_axis_variable_names( axis_variable_names )

    @property
    def argument( self ) -> Optional[ AxisExpr ]:
        res = []
        for term in self.terms:
            if term.arguments is not None:
                if len( term.arguments ) != 1:
                    raise NotImplementedError
                res.append( term.arguments[ 0 ] )
        if len( res ) > 1:
            raise NotImplementedError
        if len( res ) == 1:
            return res[ 0 ]
        return None

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
                self.terms.append( AxisTerm( 1, name ) )
            case ast.Constant( value=int( n ) ):
                self.offset += n
            case _:
                raise ValueError( f"unsupported expression: {ast.dump( node )}" )

    @staticmethod
    def _term_from_node( node, coeff: int = 1 ) -> AxisTerm:
        match node:
            case ast.Name( id=name ):
                return AxisTerm( coeff, name )
            case ast.Subscript( value=ast.Name( id=name ), slice=cmp ):
                return AxisExpr._term_from_subscript( name, cmp, coeff )
            case ast.Call( func=ast.Name( id=name ), args=args, keywords=[] ):
                return AxisExpr._term_from_call( name, args, coeff )
            case _:
                raise ValueError( f"unsupported: {ast.dump( node )}" )

    @staticmethod
    def _term_from_subscript( name: str, cmp, coeff: int = 1 ) -> AxisTerm:
        if isinstance( cmp, ast.Tuple ):
            selection = [ AxisExpr._from_node( elt ) for elt in cmp.elts ]
        else:
            selection = [ AxisExpr._from_node( cmp ) ]
        return AxisTerm( coeff, name, selection = selection )

    @staticmethod
    def _term_from_call( name: str, args, coeff: int = 1 ) -> AxisTerm:
        arguments = [ AxisExpr._from_node( arg ) for arg in args ]
        return AxisTerm( coeff, name, arguments = arguments )

    @staticmethod
    def _from_node( node ) -> AxisExpr:
        s = object.__new__( AxisExpr )
        s.offset = 0
        s.terms = []
        s._parse( node )
        return s
