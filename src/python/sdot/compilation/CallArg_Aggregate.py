from ..util.get_all_annotations import get_all_annotations

from .TemplateArgs import TemplateArgs
from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional
from weakref import ref


class _CppScope:
    """Adapts a flat `{ key: ( annotation, cpp_value ) }` map to the `_aggregate_items`
    interface, so `AxisVariableSystem` can resolve over C++-shaped tensors."""

    def __init__( self, items: dict ):
        self._items = items

    def _aggregate_items( self ):
        return self._items


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
        self._additional_signature_items = []

        # batch_axes prepended (as leading tensor dimensions) to every tensor of this aggregate.
        # Source, in priority order: the instance (vmap sets it per-instance), an explicit
        # `batch_axes` in the Return's type_kwargs, then the class default. They are *not* struct
        # members — their sizes are read from tensor shapes — so a batched aggregate stays the
        # same C++ struct as its unbatched form (e.g. Cell stays Cell, BatchOfCells = Cell + axis).
        if python_value is not None:
            self.batch_axes = list( getattr( python_value, "batch_axes", [] ) )
        elif ctor_kwargs and "batch_axes" in ctor_kwargs:
            self.batch_axes = list( ctor_kwargs[ "batch_axes" ] )
        else:
            self.batch_axes = list( getattr( python_class, "batch_axes", [] ) )

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
                # prepend this aggregate's batch_axes as leading dims of each tensor annotation
                field_batch_axes = None
                if self.batch_axes and ( make_variant := getattr( annotation, "make_variant", None ) ):
                    annotation = make_variant( self.batch_axes, 0 )
                    # the prepended axes become this field's named batch axes, at leading positions
                    field_batch_axes = [ ( axis, i ) for i, axis in enumerate( self.batch_axes ) ]
                result = CallArg.factory( call_args, self, name, annotation, value, io_category, ctor_args, ctor_kwargs )
                if result is None:
                    self._additional_signature_items.append( f"{ name }_absent" )
                else:
                    # record per-tensor batch axes ( name, position ) so the named C++ machinery
                    # ( AxisNames / BatchOf ) can address them; the aggregate's batch axes are a
                    # leading dim shared by every tensor field. ( None on a field means: not a
                    # batched tensor field — e.g. a nested aggregate, which carries its own. )
                    if field_batch_axes is not None and getattr( result, "batch_axes", "missing" ) is None:
                        result.batch_axes = field_batch_axes
                    self.sub_dict[ name ] = result

    def _field_batch_tags( self, argument ) -> Optional[ str ]:
        """`ax_…,ax_…` axis-name tags to wrap `argument` in a `BatchOf<…>`, or None.

        A *raw* member ( tensor or dynamic axis ) is wrapped in `BatchOf` exactly when it carries
        batch axes *and* this aggregate is itself unbatched (the Parameters root): such a member is
        the body-facing object the `parallel_over` loop indexes by a named `batch_index`, and a raw
        view's own `operator()` would otherwise consume the index positionally. A *batched
        aggregate* member ( e.g. a Cell field ) is never wrapped — it carries its own named
        `operator()`. A member of a batched aggregate is never wrapped either ( its axes are
        resolved by the aggregate's `squeeze` )."""
        if self.batch_axes:
            return None
        if getattr( argument, "sub_dict", None ) is not None:  # a batched aggregate: not wrapped
            return None
        ax = getattr( argument, "batch_axes", None )
        if ax is None:                                         # not declared batched: a plain field
            return None
        # declared via Batched ( None vs [] is meaningful ): wrap even when empty ( []  = broadcast,
        # an empty BatchOf whose operator() returns the field unchanged ). `( name, position )` pairs.
        names = [ name for name, _pos in sorted( ax, key = lambda np: np[ 1 ] ) ]
        return ",".join( f"ax_{ name }" for name in names )

    @property
    def children( self ) -> dict[ str, 'CallArg' ]:
        """Override: an aggregate's sub-arguments are its analysed attributes."""
        return self.sub_dict

    def signature( self ):
        """(analysis, override) Per-binding signature: the concatenated signatures of the children."""
        lst = []
        for name, attr in self.sub_dict.items():
            lst.append( f"{ name }_{ attr.signature() }" )
        lst.extend( self._additional_signature_items )
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
            parameters_template_macro = f"#define TEMPLATE_PARAMETERS_OF_{ cpp_name } template<{ params }>"
            parameters_declaration_macro = f"#define PARAMETERS_DECLARATION_OF_{ cpp_name } { params }"
            parameter_names_macro = f"#define PARAMETER_NAMES_OF_{ cpp_name } { names }"
        else:
            parameters_template_macro = f"#define TEMPLATE_PARAMETERS_OF_{ cpp_name }"
            parameters_declaration_macro = f"#define PARAMETERS_DECLARATION_OF_{ cpp_name }"
            parameter_names_macro = f"#define PARAMETER_NAMES_OF_{ cpp_name }"

        # ATTRIBUTES_OF_<Name>: struct body as a multi-line macro (backslash continuations)
        if body_lines:
            macro_body       = "\n".join( ( line or "    " ) + " \\" for line in body_lines[ :-1 ] ) + "\n" + ( body_lines[ -1 ] or "    " )
            attributes_macro = f"#define ATTRIBUTES_OF_{ cpp_name } \\\n{ macro_body }"
        else:
            attributes_macro = f"#define ATTRIBUTES_OF_{ cpp_name }"

        # batch-axis tag structs this struct's body refers to ( e.g. its named operator() ); they
        # must be visible where the ATTRIBUTES macro expands, so declare them in this header.
        from .axis_tag_decls import axis_tag_decls
        tag_lines = axis_tag_decls( self.batch_axes )

        all_lines = [ "#pragma once", "" ] + inc_lines + [ "" ] + tag_lines + [ "", parameters_template_macro, "", parameters_declaration_macro, "", parameter_names_macro, "", attributes_macro ]
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

    def _aggregate_items( self ):
        """(analysis) Duck-typed view consumed by `AxisVariableSystem`: one item per direct
        child, pairing the child itself (exposing the declared `.shape` for tensors) with its
        concrete python value (exposing the numpy `.shape`). Nested aggregates and shapeless
        leaves have no `.shape` and are simply skipped by the resolver."""
        res = {}
        for name, child in self.sub_dict.items():
            res[ name ] = ( child, getattr( child, "python_value", None ) )
        return res

    def check_axis_consistency( self ):
        """(analysis, override) Check this scope's tensors agree, then recurse into children."""
        from ..aggregate.AxisVariableSystem import AxisVariableSystem
        AxisVariableSystem( self ).check_consistency()
        for argument in self.sub_dict.values():
            argument.check_axis_consistency()

    # -- C++ codegen resolution (shapes as CppScalar) -------------------------

    def axis_system( self, use_attributes: bool = False, recursive: bool = True, explicit_values = None ):
        """
        (code generation) `AxisVariableSystem` over this scope's tensors with C++-expression
        shapes (`CppScalar`), so `value_of` returns the C++ value of an axis variable.

        `recursive`      also flatten nested aggregates' tensors (in scope as flat FFI
                         variables during C++ assembly); when False, only direct tensors.
        `use_attributes` reference tensors by attribute path instead of flat FFI name.
        """
        from ..aggregate.AxisVariableSystem import AxisVariableSystem

        items: dict = {}
        for name, argument in self.sub_dict.items():
            argument._collect_cpp_axis_items( items, [ name ], use_attributes, recursive )
        return AxisVariableSystem( _CppScope( items ), explicit_values = explicit_values )

    def _collect_cpp_axis_items( self, items, attributes, use_attributes, recursive ):
        """(code generation, override) A nested aggregate is a distinct scope: its tensors are
        contributed only when `recursive` (C++ assembly, where every tensor is a flat FFI var)."""
        if not recursive:
            return
        for name, argument in self.sub_dict.items():
            argument._collect_cpp_axis_items( items, attributes + [ name ], use_attributes, recursive )

    def cpp_runtime_expr( self, name: str, explicit_values = None ) -> str:
        """(code generation) C++ expression for the run-time value of axis variable `name`."""
        val = self.axis_system( explicit_values = explicit_values ).value_of( name )
        return str( val ) if val is not None else "0"

    # capacity of a DynamicAxis resolves exactly like a plain axis
    cpp_capacity_expr = cpp_runtime_expr

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

        ct_axis_variable_names : list[ tuple ] = []
        axis_variable_names: list[ str ] = []
        template_args = TemplateArgs()
        includes = set()
        for name, argument in self.sub_dict.items():
            argument.get_ct_axis_variable_names( ct_axis_variable_names, [ name ] )
            argument.get_axis_variable_names( axis_variable_names )
            argument.get_template_args( template_args, [ name ] )
            argument.get_includes( includes )

        lines = []

        # apply_values, batch_sizes() + operator()()
        # batch_axes are leading tensor dimensions (not struct members): their sizes are read from
        # the shape of a representative field, so a batched aggregate keeps the same struct layout.
        batch_axes = self.batch_axes
        rep_field  = next( iter( self.sub_dict ), None )
        batch_size_exprs = [ f"{ rep_field }.shape( Ct<int,{ i }>() )" for i in range( len( batch_axes ) ) ]
        lines.append(  f"    template<class F> HD auto apply_values( F &&func ) const {{ return func( { ', '.join( self.sub_dict.keys() ) } ); }}" )
        lines.append(  f"    template<class F> HD auto apply_values( F &&func ) {{ return func( { ', '.join( self.sub_dict.keys() ) } ); }}" )
        lines.append(  f"    HD auto batch_sizes() const {{ return tuple( { ', '.join( batch_size_exprs ) } ); }}" )

        # Generic slice operator: Tuple<> → identity, any other index → slice each field.
        # Each field is sliced via its own operator(), so the resulting struct has different
        # (lower-rank) tensor types. We spell out the return type's template arguments explicitly
        # (decltype of each sliced field) because aggregate CTAD from designated initializers is
        # C++20-only — C++17 cannot deduce them. The return type's template-parameter order matches
        # this struct's `template_args` (same fields, the batch axis is a runtime member, not a param).
        # When an unbatch_version exists (e.g. BatchOfCells → Cell) it is used as the return type;
        # otherwise the same struct type is returned (Cell → Cell, for vmap).
        if unbatch_version is not None:
            includes.add( f"sdot/{ unbatch_version.__name__ }.h" )

        def _rebuilt_struct( field_proj, accessor_tmpl: str, return_type: str ) -> None:
            """Emit a `return ReturnType{ .axes…, .field = field_proj( arg, name ), … };` body.

            `field_proj( argument, name )` projects each direct member (its own slice/squeeze, so a
            scalar parameter passes through). The return type's `T_…` template args are deduced via
            `accessor_tmpl` applied to the (possibly underscore-flattened) member accessor — this is
            only well-formed for a flat struct (the batched aggregates that actually use it); the
            unbatched-root `operator()` is never instantiated, so its latent form is harmless."""
            rt_args = []
            for n, _ in template_args:
                if n.startswith( "T_" ):  # a tensor/aggregate field; its (depth-1) accessor is n[2:]
                    rt_args.append( f"decltype( { accessor_tmpl.format( acc = n[ 2: ] ) } )" )
                else:  # TI, ct_<axis>: passed through unchanged
                    rt_args.append( n )
            if rt_args:
                return_type += f"<{ ', '.join( rt_args ) }>"
            lines.append( f"        return { return_type }{{" )
            for axis_variable_name in axis_variable_names:
                if axis_variable_name not in batch_axes:
                    lines.append( f"            .{ axis_variable_name } = { axis_variable_name }," )
            for name, argument in self.sub_dict.items():
                lines.append( f"            .{ name } = { field_proj( argument, name ) }," )
            lines.append(  "        };" )

        if batch_axes:
            # batched aggregate: `squeeze( nm, i )` resolves one named axis at a time, squeezing it
            # across every member and rebuilding ( type-stable, lower-rank members ). The body-facing
            # `operator()( batch_index )` peels every axis present in the index by name, reusing that
            # squeeze — so the aggregate stays a real aggregate ( p.cell.dim, p.cell.batch_sizes()
            # keep working ) instead of being hidden behind a BatchOf wrapper.
            includes.add( "sdot/support/containers/BatchOf.h" )
            ax_tags = ",".join( f"ax_{ name }" for name in batch_axes )
            lines.append(  "    template<class Name> HD auto squeeze( Name nm, PI i ) const {" )
            _rebuilt_struct( lambda argument, name: argument.named_squeeze_code( name ), "{acc}.squeeze( nm, i )", base_cpp_name )
            lines.append(  "    }" )
            lines.append(  "    template<class BI> HD auto operator()( BI batch_index ) const {" )
            lines.append( f"        return peel_named_axes( *this, batch_index, container_tags::AxisNames<{ ax_tags }>{{}} );" )
            lines.append(  "    }" )
        else:
            # unbatched aggregate ( e.g. the Parameters root ): the only projection is the empty
            # index identity ( the batched members are wrapped in BatchOf, indexed directly ).
            return_type = unbatch_version.__name__ if unbatch_version is not None else base_cpp_name
            lines.append(  "    template<class BI> HD auto operator()( BI batch_index ) const {" )
            lines.append(  "        if constexpr ( std::is_same_v<std::decay_t<decltype( batch_index )>, Tuple<>> ) {" )
            lines.append(  "            return *this;" )
            lines.append(  "        } else {" )
            _rebuilt_struct( lambda argument, name: argument.batch_slice_code( name, "batch_index" ), "{acc}( batch_index )", return_type )
            lines.append(  "        }" )
            lines.append(  "    }" )

        lines.append(  "" )

        # axis variable values
        if axis_variable_names:
            lines.append(  "    /* axis values */" )
            for axis_variable_name in axis_variable_names:
                if axis_variable_name in batch_axes: # batch axes live as leading tensor dims, not members
                    continue
                # scope-local axes are bare; they are compile-time iff exposed as a bare ct tuple
                if ( axis_variable_name, ) in ct_axis_variable_names: # always a ct_axis -> make a constexpr
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

        # data members. A batched root member is wrapped in BatchOf<…> so the body can index it
        # by a named batch_index ( p.cell( batch_index ), p.frames( batch_index ), … ).
        lines.append(  "    /* attributes */" )
        for name, argument in self.sub_dict.items():
            member_type = argument.cpp_type_name( [ name ] )
            if ( tags := self._field_batch_tags( argument ) ) is not None:
                includes.add( "sdot/support/containers/BatchOf.h" )
                member_type = f"BatchOf<{ member_type }{ ',' + tags if tags else '' }>"
            lines.append( f"    { member_type } { name };" )

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
        ct_axis_variable_names : list[ tuple( str ) ] = []
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

        # axes. compile-time axes have a known int value (emitted as a constexpr); seed them as
        # explicit values so a run-time axis expression depending on them resolves to that int.
        ct_values = { "_".join( n ): self.value_of_axis_variable( n ) for n in ct_axis_variable_names }
        for axis_variable_name in axis_variable_names:
            if axis_variable_name in self.batch_axes: # batch axes are leading tensor dims, not members
                continue
            # scope-local axes are bare; they are compile-time iff exposed as a bare ct tuple
            if ( axis_variable_name, ) in ct_axis_variable_names: # always a ct_axis -> make a constexpr
                lines.append( f"{ beg_line }    .{ axis_variable_name } = Ct<TI,{ ct_values[ axis_variable_name ] }>()," )
            else: # else, attribute to be filled during construction
                lines.append( f"{ beg_line }    .{ axis_variable_name } = { self.cpp_runtime_expr( axis_variable_name, explicit_values = ct_values ) }," )

        # attributes. A batched root member is wrapped in BatchOf<…> ( its type is deduced from the
        # value so we never respell it ) to match the struct's member type.
        for name, argument in self.sub_dict.items():
            value = argument.assembled_code( beg_line + '    ' )
            if ( tags := self._field_batch_tags( argument ) ) is not None:
                value = f"BatchOf<DECAYED_TYPE_OF( ( { value } ) ){ ',' + tags if tags else '' }>{{ { value } }}"
            lines.append( f"{ beg_line }    .{ name } = { value }," )

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

        res._additional_signature_items = list( self._additional_signature_items )

        for name, attr in self.sub_dict.items():
            res.sub_dict[ name ] = attr.backward_version( call_args, driver, outputs, grads_of_the_outputs, res, differentiable_inputs )

        return res
