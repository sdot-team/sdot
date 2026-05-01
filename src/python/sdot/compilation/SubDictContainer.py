from __future__ import annotations
from .CallArg import CallArg


class SubDictContainer:
    """
    Base for objects that own a named collection of child CallArgs.

    Responsibilities of this class:
      - hold `sub_dict` and define what it means
      - resolve axis variable values from tensor shapes (get_variable_value)
      - build the linear system that relates tensor shapes to axis variables (axis_variable_equation)
      - generate C++ struct declarations and initializers (struct_decl, assembled_code)

    Subclasses are responsible for:
      - populating `sub_dict` during construction
      - defining what to do with the child args (FFI binding, return assembly, ...)
    """

    sub_dict: dict[ str, CallArg ]
    # Maps each argument name to its CallArg.
    # Ordered: the order reflects the struct field order and the call signature.

    def get_variable_value( self, axis_name: str ) -> int:
        """
        Return the concrete integer value of an axis variable (e.g. `dim`, `nb_points`)
        by reading it from ctor_kwargs (compile-time known) or by back-computing it
        from the shapes of the tensors currently in `sub_dict`.

        Raises RuntimeError if the variable cannot be resolved.
        """
        from ..util.index import index

        axes, ct_axes = {}, {}
        for argument in self.sub_dict.values():
            if ka := getattr( argument, "ctor_kwargs", None ):
                if axis_name in ka:
                    return int( ka[ axis_name ] )
            argument.get_axes( axes, ct_axes )

        tensor_names, tensor_axes, matrix, vector = self.axis_variable_equation( axes, True )

        num_axis = index( list( axes.keys() ), axis_name )
        for num_case in range( len( tensor_names ) ):
            coeff = matrix[ num_case ][ num_axis ]
            if coeff == 0 or sum( matrix[ num_case ] != 0 ) != 1:
                continue
            val = self.sub_dict[ tensor_names[ num_case ] ].python_value
            if val is not None:
                res = val.shape[ tensor_axes[ num_case ] ]
                return ( res - int( vector[ num_case ] ) ) // int( coeff )

        raise RuntimeError( f"Unable to find axis variable value for '{ axis_name }'" )

    def axis_variable_equation( self, axes: dict, use_attributes: bool ):
        """
        Build the linear system that relates tensor shapes to axis variable values.

        Each equation i says:  shape[ tensor_axes[i] ] of tensor_names[i]
                               == matrix[i] · axis_values + vector[i]

        Returns ( tensor_names, tensor_axes, matrix, vector ).
        """
        import numpy

        tensor_names, tensor_axes, matrix, vector = [], [], [], []
        for name, argument in self.sub_dict.items():
            argument.get_all_the_ways_to_get(
                list( axes.keys() ), [ name ], use_attributes,
                tensor_names, tensor_axes, matrix, vector
            )

        return tensor_names, tensor_axes, numpy.array( matrix ), numpy.array( vector )

    @staticmethod
    def get_axis_variable( axes: dict, axis_name: str, axis_selection, tensor_names, tensor_axes, matrix, vector ) -> str:
        """
        Return C++ expression that evaluates to the value of `axis_name` at runtime,
        by picking the first tensor whose single non-zero coefficient matches this axis.

        `axis_selection` is None for plain axes and a list for dynamic (max_of_*) axes.
        Returns a call to `first_positive(...)` over all candidate tensors.
        """
        from ..util.index import index

        cases = []
        num_axis = index( list( axes.keys() ), axis_name )
        for num_case in range( len( tensor_names ) ):
            coeff = matrix[ num_case ][ num_axis ]
            if coeff == 0 or sum( matrix[ num_case ] != 0 ) != 1:
                continue
            op = f"{ tensor_names[ num_case ] }.size( { tensor_axes[ num_case ] } )"
            if vector[ num_case ]:
                if coeff != 1:
                    op = f"( { op } - { vector[ num_case ] } ) / { coeff }"
                else:
                    op = f"{ op } - { vector[ num_case ] }"
            elif coeff != 1:
                op = f"{ op } / { coeff }"
            cases.append( f"{ tensor_names[ num_case ] }.is_valid() ? { op } : -1" )

        return f"first_positive( { ', '.join( cases ) } )"

    def struct_decl( self, base_cpp_name: str, includes: set, lines: list[ str ], end_lines: list[ str ] = [] ) -> str:
        """
        Generate the C++ struct declaration for this container's sub_dict.

        Appends to `lines` and `includes` in place; also returns the full source as a string
        (includes + lines + end_lines joined by newlines).
        """
        template_args : dict[ str, str ] = {}
        ct_axes       : dict[ str, int ] = {}
        axes          : dict             = {}

        for name, argument in self.sub_dict.items():
            argument.get_template_args( template_args, [ name ] )
            argument.get_includes( includes )
            argument.get_axes( axes, ct_axes )

        for ct_axis_name in ct_axes:
            template_args[ f"ct_{ ct_axis_name }_value" ] = "int"

        lines.append( f"template<{ ', '.join( f'{ t } { n }' for n, t in template_args.items() ) }>" )
        lines.append( f"struct { base_cpp_name } {{" )

        # axis accessor methods
        if axes:
            tensor_names, tensor_axes, matrix, vector = self.axis_variable_equation( axes, True )
            for axis_name, axis_selection in axes.items():
                dv = SubDictContainer.get_axis_variable( axes, axis_name, axis_selection, tensor_names, tensor_axes, matrix, vector )
                complete_name = f"max_of_{ axis_name }" if axis_selection is not None else axis_name
                ct_code = f"if constexpr ( ct_{ axis_name }_value >= 0 ) return ct_{ axis_name }; else " if axis_name in ct_axes else ""
                lines.append( f"    auto { complete_name }() const {{ { ct_code }return { dv }; }}" )
            lines.append( "" )

        # compile-time axis members
        for ct_axis_name in ct_axes:
            lines.append( f"    CtInt<ct_{ ct_axis_name }_value> ct_{ ct_axis_name };" )

        # data members
        for name, argument in self.sub_dict.items():
            lines.append( f"    { argument.cpp_type_name( [ name ] ) } { name };" )

        lines.append( "};" )
        lines.extend( end_lines )

        return "\n".join( [ "#pragma once" ] + [ f"#include <{ inc }>" for inc in includes ] + lines )

    def assembled_code( self, struct_name: str, beg_line: str ) -> str:
        """
        Generate the C++ initializer for this container as a `struct_name{ .field = ..., }` literal.

        `beg_line` is the indentation prefix for nested lines.
        Subclasses that implement the `CallArg` interface override this with a one-argument
        version that supplies `struct_name` automatically (e.g. from `base_cpp_name()`).
        """
        lines = [ struct_name + "{" ]

        ct_axes : dict[ str, int ] = {}
        axes    : dict             = {}
        for argument in self.sub_dict.values():
            argument.get_axes( axes, ct_axes )

        for ct_axis_name in ct_axes:
            lines.append( f"{ beg_line }    .ct_{ ct_axis_name } = CtInt<{ self.get_variable_value( ct_axis_name ) }>()," )

        for name, argument in self.sub_dict.items():
            lines.append( f"{ beg_line }    .{ name } = { argument.assembled_code( beg_line + '    ' ) }," )

        lines.append( beg_line + "}" )
        return "\n".join( lines )
