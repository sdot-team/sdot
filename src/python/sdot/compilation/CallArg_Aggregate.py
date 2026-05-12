# from sdot.compilation.CallArgsAnalysis import CallArgsAnalysis
from ..util.get_all_annotations import get_all_annotations

from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import TYPE_CHECKING
from weakref import ref

class CallArg_Aggregate( CallArg ):
    """ input or mutable
    """

    if TYPE_CHECKING:
        sub_dict : dict[ str, CallArg ] #

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        res = CallArg_Aggregate()

        # CallArg attributes
        res.name_in_parent = name_in_parent
        res.parent = ref( parent ) if parent is not None else None

        res.python_class = python_class
        res.python_value = python_value

        res.io_category = io_category

        res.ctor_kwargs = ctor_kwargs
        res.ctor_args = ctor_args

        # sub_dict
        res.sub_dict = {}
        for name, annotation in get_all_annotations( python_class ).items():
            value = None
            if python_value is not None:
                try:
                    value = getattr( python_value, name )
                except AttributeError:
                    raise RuntimeError( f"Unable to find attribute { name } in { python_value }" )
            res.sub_dict[ name ] = CallArg.factory( call_args, res, name, annotation, value, io_category, ctor_args, ctor_kwargs )

        return res

    def signature( self ):
        lst = []
        for name, attr in self.sub_dict.items():
            lst.append( f"{ name }_{ attr.signature() }" )
        return "__".join( lst )

    def get_template_args( self, template_args, names ):
        for name, attr in self.sub_dict.items():
            attr.get_template_args( template_args, names + [ name ] )

    def cpp_type_name( self, names ):
        from .TemplateArgs import TemplateArgs

        template_args = TemplateArgs()
        self.get_template_args( template_args, names )

        if self.python_value:
            res = self.python_value.__class__.__name__
        else:
            res = self.python_class.__name__

        if template_args:
            res += f"<{ ','.join( n for n, _ in template_args ) }>"

        return res

    def base_cpp_name( self ) -> str:
        return self.python_class.__name__

    def assemble_return( self ):
        ctor_args = {}
        for name, call_arg in self.sub_dict.items():
            ctor_args[ name ] = call_arg.assemble_return()
        return self.python_class( **ctor_args )

    def get_includes( self, includes: set ):
        includes.add( f"sdot/{ self.base_cpp_name() }.h" )

    def generate_structure( self, already_visited ):
        if self.python_class in already_visited:
            return
        already_visited.add( self.python_class )

        # unbatch
        bv = getattr( self.python_class, "BaseVersion", None )
        if bv == self.python_class:
            bv = None

        if bv:
            io = IoCategory( want_output = False, want_return = False, has_input = False )
            unbatch_call_arg = CallArg_Aggregate.factory( call_args = None, parent = None, name_in_parent = "", python_class = bv, python_value = None, io_category = io, ctor_args = [], ctor_kwargs = {} )
            unbatch_call_arg.generate_structures( already_visited )

        body_lines, includes, template_args = self.struct_body( self.base_cpp_name(), unbatch_version = bv )
        includes.add( "sdot/support/DynamicAxis.h" )

        cpp_name = self.base_cpp_name()

        inc_lines = []
        for inc in sorted( includes, key = lambda s: ( -len( s ), s ) ):
            if inc.startswith( "." ):
                inc_lines.append( f"#include \"{ inc }\"" )
            else:
                inc_lines.append( f"#include <{ inc }>" )

        # PARAMETERS_OF_<Name>: template signature for the outer hand-written struct
        if template_args:
            params = ', '.join( f'{ ta.cpp_type } { n }' for n, ta in template_args )
            names = ', '.join( n for n, _ in template_args )
            parameters_declaration_macro = f"#define PARAMETERS_DECLARATION_OF_{ cpp_name } template<{ params }>"
            parameter_names_macro = f"#define PARAMETER_NAMES_OF_{ cpp_name } { names }"
        else:
            parameters_declaration_macro = f"#define PARAMETERS_DECLARATION_OF_{ cpp_name }"
            parameter_names_macro = f"#define PARAMETER_NAMES_OF_{ cpp_name }"

        # ATTRIBUTES_OF_<Name>: struct body as a multi-line macro (backslash continuations)
        if body_lines:
            macro_body       = "\n".join( ( line or "    " ) + " \\" for line in body_lines[ :-1 ] ) + "\n" + ( body_lines[ -1 ] or "    " )
            attributes_macro = f"#define ATTRIBUTES_OF_{ cpp_name } \\\n{ macro_body }"
        else:
            attributes_macro = f"#define ATTRIBUTES_OF_{ cpp_name }"

        all_lines = [ "#pragma once", "" ] + inc_lines + [ "", parameters_declaration_macro, "", parameter_names_macro, "", attributes_macro ]
        code = "\n".join( all_lines )

        from ..generated_files.compilation_directories import generated_includes_dir
        path = generated_includes_dir() / f"{ self.base_cpp_name() }.h"
        try:
            old_text = path.read_text()
        except FileNotFoundError:
            old_text = ""
        if code != old_text:
            path.write_text( code )

    def generate_structures( self, already_visited ):
        self.generate_structure( already_visited )

        for argument in self.sub_dict.values():
            argument.generate_structures( already_visited )

    def get_all_the_ways_to_get( self, axis_names, attributes, use_attributes, tensor_names, tensor_axes, matrix, vector ):
        pass

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        for argument in self.sub_dict.values():
            argument.get_arg_decl( non_differentiable_inputs, differentiable_inputs, parameters, outputs )

    def get_variable_value( self, axis_name: str ) -> int:
        """
        Return the concrete integer value of an axis variable (e.g. `dim`, `nb_points`)
        by reading it from ctor_kwargs (compile-time known) or by back-computing it
        from the shapes of the tensors currently in `sub_dict`.

        Raises RuntimeError if the variable cannot be resolved.
        """
        from ..util.index import index

        # in argument ctor_kwargs ?
        if ka := getattr( self, "ctor_kwargs", None ):
            if axis_name in ka:
                return int( ka[ axis_name ] )

        # in argument ctor_kwargs ?
        for argument in self.sub_dict.values():
            if ka := getattr( argument, "ctor_kwargs", None ):
                if axis_name in ka:
                    return int( ka[ axis_name ] )

        # in argument properties ?
        axes, ct_axes = {}, {}
        for argument in self.sub_dict.values():
            python_value = getattr( argument, "python_value", None )
            if python_value is not None:
                value = getattr( python_value, axis_name, None )
                if value is not None:
                    return int( value )

        # in self properties ?
        python_value = getattr( self, "python_value", None )
        if python_value is not None:
            value = getattr( python_value, axis_name, None )
            if value is not None:
                return int( value )

        # else, make the system using python_value.shape
        axes, ct_axes = {}, {}
        for argument in self.sub_dict.values():
            argument.get_axes( axes, ct_axes )

        tensor_names, tensor_axes, matrix, vector = self.axis_variable_equation( axes, True )

        num_axis = index( list( axes.keys() ), axis_name )
        for num_case in range( len( tensor_names ) ):
            coeff = matrix[ num_case ][ num_axis ]
            if coeff == 0 or sum( matrix[ num_case ] != 0 ) != 1:
                continue
            if tensor_names[ num_case ] in self.sub_dict:
                val = self.sub_dict[ tensor_names[ num_case ] ].python_value
                if val is not None:
                    res = val.shape[ tensor_axes[ num_case ] ]
                    return ( res - int( vector[ num_case ] ) ) // int( coeff )

        raise RuntimeError( f"Unable to find axis variable value for '{ axis_name }' in '{ self }'" )

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

    def get_axis_variable( self, axes: dict, axis_name: str, axis_selection, tensor_names, tensor_axes, matrix, vector ) -> str:
        """
        Return C++ expression that evaluates to the value of `axis_name` at runtime,
        by picking the first tensor whose single non-zero coefficient matches this axis.

        `axis_selection` is None for plain axes and a list for dynamic (max_of_*) axes.
        Returns a call to `first_positive(...)` over all candidate tensors.
        """
        from ..util.index import index

        cases = []

        # using ct_axes
        for field in self.sub_dict.values():
            if ct_axes := getattr( field, "ct_axes", None ):
                if axis_name in ct_axes:
                    cases.append( f"ct_{ axis_name }" )

        # using shapes
        num_axis = index( list( axes.keys() ), axis_name )
        for num_case in range( len( tensor_names ) ):
            coeff = matrix[ num_case ][ num_axis ]
            if coeff == 0 or sum( matrix[ num_case ] != 0 ) != 1:
                continue
            op = f"{ tensor_names[ num_case ] }.shape( { tensor_axes[ num_case ] } )"
            if vector[ num_case ]:
                if coeff != 1:
                    op = f"( { op } - { vector[ num_case ] } ) / { coeff }"
                else:
                    op = f"{ op } - { vector[ num_case ] }"
            elif coeff != 1:
                op = f"{ op } / { coeff }"
            cases.append( f"{ tensor_names[ num_case ] }.is_valid() ? { op } : -1" )

        return f"first_positive( \"{ axis_name }\", { ', '.join( cases ) } )"

    def struct_body( self, base_cpp_name: str, unbatch_version = None ):
        """
        Generate the content of a C++ struct: methods and data members, without the
        `template<...> struct Name {` / `};` wrapper.

        Returns ( body_lines, includes, template_args ) so the caller can either wrap
        them into a full struct declaration (struct_decl) or write them to an include
        file for embedding inside a hand-written struct.
        """
        from .TemplateArgs import TemplateArgs

        template_args = TemplateArgs()
        includes      = set()
        ct_axes       : dict[ str, int ] = {}
        axes          : dict = {}

        for name, argument in self.sub_dict.items():
            argument.get_template_args( template_args, [ name ] )
            argument.get_includes( includes )
            argument.get_axes( axes, ct_axes )

        for ct_axis_name in ct_axes:
            template_args.add( f"ct_{ ct_axis_name }", "int", 0 )

        lines = []

        # row accessor (batch → single-row)
        if unbatch_version is not None:
            includes.add( f"./{ unbatch_version.__name__ }.h" )
            lines.append(  "    auto row( PI index ) const {" )
            lines.append( f"        return { unbatch_version.__name__ }{{" )
            for ct_axis_name in ct_axes:
                lines.append( f"            .ct_{ ct_axis_name }_inst = CtdInt<ct_{ ct_axis_name }>()," )
            for name, argument in self.sub_dict.items():
                lines.append( f"            .{ name } = { name }.row( index )," )
            lines.append(  "        };" )
            lines.append(  "    }" )

        # axis accessor methods
        if axes:
            # tensor_names, tensor_axes, matrix, vector = self.axis_variable_equation( axes, True )
            for axis_name, axis_selection in axes.items():
                complete_axis_name = f"max_of_{ axis_name }" if axis_selection is not None else axis_name

                if axis_name in ct_axes and ct_axes[ axis_name ] is None: # always a ct_axis -> make a constexpr
                    lines.append( f"    static constexpr SI { complete_axis_name } = ct_{ axis_name };" )
                else: # else, attribute to be filled during construction
                    lines.append( f"    SI { complete_axis_name };" )

            lines.append( "" )

        # with_same_shape
        lines.append( "    void with_same_shape( auto &&func ) const {" )
        s = "        "
        for name, argument in self.sub_dict.items():
            s = argument.beg_with_same_shape( name, s, lines )
        lines.append( s + f"{ base_cpp_name } new_value{{" )
        for ct_axis_name in ct_axes:
            lines.append( s + f"    .ct_{ ct_axis_name }_inst = CtdInt<ct_{ ct_axis_name }>()," )
        for name, argument in self.sub_dict.items():
            lines.append( s + f"    .{ name } = { name }," )
        lines.append( s + "};" )
        lines.append( s + "func( new_value );" )
        for name, argument in self.sub_dict.items():
            s = argument.end_with_same_shape( name, s, lines )
        lines.append( "    }" )

        # compile-time axis members
        for ct_axis_name in ct_axes:
            lines.append( f"    CtdInt<ct_{ ct_axis_name }> ct_{ ct_axis_name }_inst;" )

        # data members
        for name, argument in self.sub_dict.items():
            lines.append( f"    { argument.cpp_type_name( [ name ] ) } { name };" )

        return lines, includes, template_args

    def struct_decl( self, base_cpp_name: str, includes: set, lines: list[ str ], unbatch_version = None ) -> None:
        """
        Append a full C++ template struct declaration to `lines` and update `includes`.
        Used when embedding a struct inside a larger generated source file (e.g. FFI handler).
        """
        body_lines, body_includes, template_args = self.struct_body( base_cpp_name, unbatch_version )
        includes.update( body_includes )

        lines.append( f"template<{ ', '.join( f'{ ta.cpp_type } { n }' for n, ta in template_args ) }>" )
        lines.append( f"struct { base_cpp_name } {{" )
        lines.extend( body_lines )
        lines.append( "};" )

    def assembled_code( self, beg_line: str, struct_name = None ) -> str:
        """
        Generate the C++ initializer for this container as a `struct_name{ .field = ..., }` literal.

        `beg_line` is the indentation prefix for nested lines.
        Subclasses that implement the `CallArg` interface override this with a one-argument
        version that supplies `struct_name` automatically (e.g. from `base_cpp_name()`).
        """
        lines = [ self.base_cpp_name() + "{" ]

        if struct_name is None:
            struct_name = self.base_cpp_name()

        ct_axes : dict[ str, int ] = {}
        axes : dict = {}
        for argument in self.sub_dict.values():
            argument.get_axes( axes, ct_axes )

        # dynamic axes
        tensor_names, tensor_axes, matrix, vector = self.axis_variable_equation( axes, use_attributes = False )
        for axis_name, axis_selection in axes.items():
            complete_axis_name = f"max_of_{ axis_name }" if axis_selection is not None else axis_name
            dv = self.get_axis_variable( axes, axis_name, axis_selection, tensor_names, tensor_axes, matrix, vector )
            ct_code = ""
            if axis_name in ct_axes:
                if ct_axes[ axis_name ] is None:
                    continue
                ct_code = f"ct_{ complete_axis_name } >= 0 ? { complete_axis_name } : "
            lines.append( f"{ beg_line }    .{ complete_axis_name } = { ct_code }{ dv }," )

        # ct axes
        for ct_axis_name in ct_axes:
            lines.append( f"{ beg_line }    .ct_{ ct_axis_name }_inst = CtdInt<{ self.get_variable_value( ct_axis_name ) }>()," )

        for name, argument in self.sub_dict.items():
            lines.append( f"{ beg_line }    .{ name } = { argument.assembled_code( beg_line + '    ' ) }," )

        lines.append( beg_line + "}" )
        return "\n".join( lines )

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs=None ):
        res = CallArg_Aggregate()
        self.init_CallArgs_backward_version( res, parent )

        res.sub_dict = {}
        for name, attr in self.sub_dict.items():
            res.sub_dict[ name ] = attr.backward_version( call_args, driver, outputs, grads_of_the_outputs, res, differentiable_inputs )

        return res
