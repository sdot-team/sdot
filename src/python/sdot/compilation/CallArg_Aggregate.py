from ..util.get_all_annotations import get_all_annotations

from .TemplateArgs import TemplateArgs
from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional
from weakref import ref

class CallArg_Aggregate( CallArg ):
    """
    A struct-like argument: a python object whose annotated attributes are recursively
    analysed into a `sub_dict` of child CallArgs. It is the scope that owns axis variables
    (C++ exposes them here, as struct members), and the unit emitted as a C++ struct.
    """

    sub_dict : dict[ str, 'CallArg' ]

    def __init__(
        self,
        call_args      : any,
        parent         : Optional[ ref ],
        name_in_parent : Optional[ str ],
        python_class   : any,
        python_value   : Optional[ any ],
        io_category    : IoCategory,
        ctor_args      : Optional[ list ] = None,
        ctor_kwargs    : Optional[ dict ] = None,
        build_sub_dict : bool = True
    ):
        super().__init__( name_in_parent, parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )
        self.sub_dict = {}

        # analysis: recursively analyse each annotated attribute into a sub-CallArg.
        # (skipped by backward_version, which fills sub_dict with transformed children itself.)
        if build_sub_dict:
            for name, annotation in get_all_annotations( python_class ).items():
                value = None
                if python_value is not None:
                    try:
                        value = getattr( python_value, name )
                    except AttributeError:
                        raise RuntimeError( f"Unable to find attribute { name } in { python_value }" )
                self.sub_dict[ name ] = CallArg.factory( call_args, self, name, annotation, value, io_category, ctor_args, ctor_kwargs )

    @property
    def children( self ) -> dict[ str, 'CallArg' ]:
        """Override: an aggregate's sub-arguments are its analysed attributes."""
        return self.sub_dict

    def signature( self ):
        """(analysis, override) Per-binding signature: the concatenated signatures of the children."""
        lst = []
        for name, attr in self.sub_dict.items():
            lst.append( f"{ name }_{ attr.signature() }" )
        return "__".join( lst )

    def cpp_type_name( self, names ):
        """(code generation, override) C++ type of the struct, with its template arguments."""
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
        """(code generation) Name of the C++ struct (the unbatched python class name)."""
        return self.python_class.__name__

    def assemble_return( self ):
        """(value management, override) Rebuild the python object from the children's returns."""
        ctor_args = {}
        for name, call_arg in self.sub_dict.items():
            ctor_args[ name ] = call_arg.assemble_return()
        return self.python_class( **ctor_args )

    def get_includes( self, includes: set ):
        """(code generation, override) The struct needs its own generated header."""
        includes.add( f"sdot/{ self.base_cpp_name() }.h" )

    def generate_structure( self, already_visited ):
        """(code generation) Write the generated C++ header (macros) for this struct's class."""
        if self.python_class in already_visited:
            return
        already_visited.add( self.python_class )

        # unbatch
        bv = getattr( self.python_class, "BaseVersion", None )
        if bv == self.python_class:
            bv = None

        if bv:
            io = IoCategory( want_output = False, want_return = False, has_input = False )
            unbatch_call_arg = CallArg_Aggregate( call_args = None, parent = self, name_in_parent = "unbatch", python_class = bv, python_value = None, io_category = io, ctor_args = self.ctor_args if self.ctor_args is not None else [], ctor_kwargs = self.ctor_kwargs if self.ctor_kwargs is not None else {} )
            unbatch_call_arg.generate_structures( already_visited )

        body_lines, includes, template_args = self.struct_body( self.base_cpp_name(), unbatch_version = bv )
        includes.add( "sdot/support/containers/DynamicAxis.h" )

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
        code = "\n".join( all_lines ) + "\n"

        from ..generated_files.compilation_directories import generated_includes_dir
        path = generated_includes_dir() / f"{ self.base_cpp_name() }.h"
        try:
            old_text = path.read_text()
        except FileNotFoundError:
            old_text = ""
        if code != old_text:
            path.write_text( code )

    def generate_structures( self, already_visited ):
        """(code generation, override) Write this struct's header, then those of its children."""
        self.generate_structure( already_visited )

        for argument in self.sub_dict.values():
            argument.generate_structures( already_visited )

    def add_axis_tensor_sources( self, sources, attributes, use_attributes, recursive ):
        """(analysis, override) Contribute the children's tensors to an AxisVariableSystem.

        A nested aggregate is a distinct axis variable scope, so its tensors are only
        added when `recursive` is set (i.e. for C++ assembly, where every tensor is in
        scope as a flat FFI variable).
        """
        if not recursive:
            return
        for name, argument in self.sub_dict.items():
            argument.add_axis_tensor_sources( sources, attributes + [ name ], use_attributes, recursive )

    def axis_system( self, use_attributes: bool = False, recursive: bool = True ):
        """
        (analysis) Build the `AxisVariableSystem` for this aggregate, one source per tensor.

        `recursive`      include tensors of nested aggregates (used for C++ assembly where
                         every tensor is in scope as a flat FFI variable); when False, only
                         the directly contained tensors are considered.
        `use_attributes` reference tensors by attribute path instead of flat FFI name.
        """
        from ..aggregate.AxisVariableSystem import AxisVariableSystem

        sources = []
        for name, argument in self.sub_dict.items():
            argument.add_axis_tensor_sources( sources, [ name ], use_attributes, recursive )
        return AxisVariableSystem( sources )

    def check_axis_consistency( self ):
        """(analysis, override) Check this scope's tensors agree, then recurse into children."""
        self.axis_system( recursive = False ).check_consistency()
        for argument in self.sub_dict.values():
            argument.check_axis_consistency()

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        """(code generation, override) Gather the children's handler argument declarations."""
        for argument in self.sub_dict.values():
            argument.get_arg_decl( non_differentiable_inputs, differentiable_inputs, parameters, outputs )

    def struct_body( self, base_cpp_name: str, unbatch_version = None ):
        """
        (code generation) Generate the content of a C++ struct: methods and data members, without the
        `template<...> struct Name {` / `};` wrapper.

        Returns ( body_lines, includes, template_args ) so the caller can either wrap
        them into a full struct declaration (struct_decl) or write them to an include
        file for embedding inside a hand-written struct.
        """

        ct_axis_variable_names : list[ str ] = []
        axis_variable_names: list[ str ] = []
        template_args = TemplateArgs()
        includes = set()
        for name, argument in self.sub_dict.items():
            argument.get_ct_axis_variable_names( ct_axis_variable_names, [ name ] )
            argument.get_axis_variable_names( axis_variable_names )
            argument.get_template_args( template_args, [ name ] )
            argument.get_includes( includes )

        lines = []

        # batch_sizes() + slice()
        batch_axes = getattr( self.python_class, 'batch_axes', [] )
        lines.append(  "    /* batch methods */" )
        lines.append(  f"    HD auto batch_sizes() const {{ return tuple( { ', '.join( batch_axes ) } ); }}" )
        lines.append(  "    HD auto operator()( Tuple<> ) const { return *this; }" )

        # slice accessor (scalar index: batch → single-row)
        if unbatch_version is not None:
            includes.add( f"sdot/{ unbatch_version.__name__ }.h" )

            assert len( batch_axes )
            lines.append( f"    HD auto operator()( Tuple<{ ','.join( [ 'TI' ] * len( batch_axes ) ) }> batch_index ) const {{" )
            lines.append( f"        return { unbatch_version.__name__ }<PARAMETER_NAMES_OF_{ unbatch_version.__name__ }>{{" )
            # axes
            for axis_variable_name in axis_variable_names:
                if axis_variable_name not in batch_axes:
                    lines.append( f"            .{ axis_variable_name } = { axis_variable_name }," )
            # attributes
            for name, argument in self.sub_dict.items():
                lines.append( f"            .{ name } = { name }( batch_index )," )
            lines.append(  "        };" )
            lines.append(  "    }" )


            # lines.append(  "    auto operator()( auto bi ) const { return slice( PI( bi[ Ct<int,0>() ] ) ).slice( bi.without_index( Ct<int,0>() ) ); }" )

        lines.append(  "" )

        # axis variable values
        if axis_variable_names:
            lines.append(  "    /* axis values */" )
            for axis_variable_name in axis_variable_names:
                if axis_variable_name in ct_axis_variable_names: # always a ct_axis -> make a constexpr
                    lines.append( f"    Ct<TI,ct_{ axis_variable_name }> { axis_variable_name };" )
                else: # else, attribute to be filled during construction
                    lines.append( f"    SI { axis_variable_name };" )
            lines.append( "" )

        # with_same_shape
        # lines.append( "    void with_same_shape( auto &&func ) const {" )
        # s = "        "
        # for name, argument in self.sub_dict.items():
        #     s = argument.beg_with_same_shape( name, s, lines )
        # lines.append( s + f"{ base_cpp_name } new_value{{" )
        # for ct_axis_name in ct_variables:
        #     lines.append( s + f"    .ct_{ ct_axis_name } = Ct<TI,ct_{ ct_axis_name }_value>()," )
        # for name, argument in self.sub_dict.items():
        #     lines.append( s + f"    .{ name } = { name }," )
        # lines.append( s + "};" )
        # lines.append( s + "func( new_value );" )
        # for name, argument in self.sub_dict.items():
        #     s = argument.end_with_same_shape( name, s, lines )
        # lines.append( "    }" )

        # data members
        lines.append(  f"    /* attributes */" )
        for name, argument in self.sub_dict.items():
            lines.append( f"    { argument.cpp_type_name( [ name ] ) } { name };" )

        return lines, includes, template_args

    def struct_decl( self, base_cpp_name: str, includes: set, lines: list[ str ], unbatch_version = None ) -> None:
        """
        (code generation) Append a full C++ template struct declaration to `lines` and update `includes`.
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
        (code generation, override) Generate the C++ initializer for this container as a `struct_name{ .field = ..., }` literal.

        `beg_line` is the indentation prefix for nested lines.
        Subclasses that implement the `CallArg` interface override this with a one-argument
        version that supplies `struct_name` automatically (e.g. from `base_cpp_name()`).
        """

        # get info
        ct_axis_variable_names : list[ str ] = []
        axis_variable_names: list[ str ] = []
        template_args = TemplateArgs()
        for name, argument in self.sub_dict.items():
            argument.get_ct_axis_variable_names( ct_axis_variable_names, [ name ] )
            argument.get_axis_variable_names( axis_variable_names )
            argument.get_template_args( template_args, [ name ] )

        # struct_name
        if struct_name is None:
            struct_name = self.base_cpp_name()
        if template_args:
            struct_name += f"<{ ','.join( t.value for _, t in template_args ) }>"

        # decl
        lines = [ struct_name + "{" ]

        # axes
        system = self.axis_system( use_attributes = False, recursive = True )
        for axis_variable_name in axis_variable_names:
            if axis_variable_name in ct_axis_variable_names: # always a ct_axis -> make a constexpr
                lines.append( f"{ beg_line }    .{ axis_variable_name } = Ct<TI,{ self.value_of_axis_variable( axis_variable_name ) }>()," )
            else: # else, attribute to be filled during construction
                lines.append( f"{ beg_line }    .{ axis_variable_name } = { system.cpp_runtime_expr( axis_variable_name ) }," )

        # attributes
        for name, argument in self.sub_dict.items():
            lines.append( f"{ beg_line }    .{ name } = { argument.assembled_code( beg_line + '    ' ) }," )

        lines.append( beg_line + "}" )

        return "\n".join( lines )

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs = None ):
        """(analysis, override) Mirror this struct for the backward call, transforming each child."""
        res = CallArg_Aggregate(
            call_args      = call_args,
            parent         = parent,
            name_in_parent = self.name_in_parent,
            python_class   = self.python_class,
            python_value   = self.python_value,
            io_category    = IoCategory.pure_input(),
            ctor_args      = self.ctor_args,
            ctor_kwargs    = self.ctor_kwargs,
            build_sub_dict = False
        )

        for name, attr in self.sub_dict.items():
            res.sub_dict[ name ] = attr.backward_version( call_args, driver, outputs, grads_of_the_outputs, res, differentiable_inputs )

        return res
