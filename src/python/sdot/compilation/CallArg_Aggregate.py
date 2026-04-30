from ..util.get_all_annotations import get_all_annotations
from ..util.index import index
from .CallArg import CallArg
# from typing import Optional

class CallArg_Aggregate( CallArg ):
    """ input or mutable
    """

    python_class: any #
    python_value: any #
    io_category : int #
    sub_dict    : dict[ CallArg ] #

    @staticmethod
    def factory( python_class, python_value, io_category: int ):
        res = CallArg_Aggregate()

        res.python_class = python_class
        res.python_value = python_value
        res.io_category = io_category
        res.sub_dict = {}

        for name, annotation in get_all_annotations( python_class ).items():
            value = None
            if python_value is not None:
                value = getattr( python_value, name )
            res.sub_dict[ name ] = CallArg.factory( annotation, value, io_category )

        return res

    def signature( self ):
        lst = []
        for name, attr in self.sub_dict.items():
            lst.append( f"{ name }_{ attr.signature() }" )
        return "__".join( lst )

    def get_template_args( self, template_args ):
        for name, attr in self.sub_dict.items():
            attr.get_template_args( template_args )

    def cpp_type_name( self, main_list ):
        template_args = {}
        self.get_template_args( template_args )

        res = self.python_value.__class__.__name__
        if len( template_args ):
            res += f"<{ ','.join( template_args ) }>"

        return res

    def base_cpp_name( self ) -> str:
        return self.python_class.__name__

    def get_includes( self, includes: set ):
        includes.add( f"sdot/generated_includes/{ self.base_cpp_name() }.h" )

    def generate_structure( self ):
        includes = set( [ "sdot/support/DynamicAxis.h" ] )
        lines = [ "", "namespace sdot {", "" ]

        code = CallArg_Aggregate.get_code( self.base_cpp_name(), self.sub_dict, includes, lines )

        #
        from ..generated_files.compilation_directories import generated_includes_dir
        path = generated_includes_dir() / f"{ self.base_cpp_name() }.h"
        try:
            old_text = path.read_text()
        except FileNotFoundError:
            old_text = ""
        if code != old_text:
            path.write_text( code )

    @staticmethod
    def get_code( base_cpp_name, sub_dict : dict[ CallArg ], includes: list[ str ], lines : list[ str ] ):
        template_args : dict[ str ] = {}
        ct_axes : dict[ int ] = {}
        axes = {}
        for argument in sub_dict.values():
            argument.get_template_args( template_args )
            argument.get_includes( includes )
            argument.get_axes( axes, ct_axes )

        for ct_axis_name in ct_axes.keys():
            template_args[ f"ct_{ ct_axis_name }_value" ] = "int"

        lines.append( f"template<{ str.join( ",", [ f"{ t } { n }" for n, t in template_args.items() ] ) }>" )
        lines.append( f"struct { base_cpp_name } {{" )

        # axis dim methods ---------------------------------------------------------------------------------------------
        if len( axes ):
            tensor_names = []
            tensor_axes = []
            matrix = []
            vector = []
            for name, argument in sub_dict.items():
                argument.get_all_the_ways_to_get( list( axes.keys() ), [ name ], tensor_names, tensor_axes, matrix, vector )
            import numpy
            matrix = numpy.array( matrix )
            vector = numpy.array( vector )

            # infox( axes.keys(), tensor_names, tensor_axes, matrix, vector )

            for axis_name, axis_selection in axes.items():
                cases = []
                num_axis = index( list( axes.keys() ), axis_name )
                for num_case in range( len( tensor_names ) ):
                    coeff = matrix[ num_case ][ num_axis ]
                    if coeff == 0 or sum( matrix[ num_case ] != 0 ) != 1:
                        continue
                    op = f"{ tensor_names[ num_case ] }[ { tensor_axes[ num_case ] } ]"
                    if vector[ num_case ]:
                        if coeff != 1:
                            op = f"( { op } - { vector[ num_case ] } ) / { coeff }"
                        else:
                            op = f"{ op } - { vector[ num_case ] }"
                    elif coeff != 1:
                        op = f"{ op } / { coeff }"
                    cases.append( f"{ tensor_names[ num_case ] }.is_valid() ? { op } : -1" )

                complete_axis_name = axis_name
                if axis_selection is not None:
                    complete_axis_name = f"max_of_{ complete_axis_name }"

                ct_code = ""
                if axis_name in ct_axes:
                    ct_code = f"if constexpr ( ct_{ axis_name }_value >= 0 ) return ct_{ axis_name }; else "

                lines.append( f"    auto { complete_axis_name }() const {{ { ct_code }return first_positive( { ", ".join( cases ) } ); }}" )

            lines.append( "" )

        # base attributes ---------------------------------------------------------------------------------------------
        for name, argument in sub_dict.items():
            lines.append( f"    { argument.cpp_type_name( None ) } { name };" )

        # dynamic axis ------------------------------------------------------------------------------------------------
        for axis_name, axis_selection in axes.items():
            if axis_selection is not None:
                lines.append( f"    DynamicAxis<{ len( axis_selection ) },Arch> { axis_name };" )

        # ct axis ------------------------------------------------------------------------------------------------
        for ct_axis_name in ct_axes.keys():
            lines.append( f"    CtInt<ct_{ ct_axis_name }_value> ct_{ ct_axis_name };" )

        # en of the struct --------------------------------------------------------------------------------------------
        lines.append( "};" )

        return "\n".join( [ f"#include <{ include }>" for include in includes ] + lines )

    def generate_structures( self ):
        self.generate_structure()

        for argument in self.sub_dict.values():
            argument.generate_structures()

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        for argument in self.sub_dict.values():
            argument.get_axes( axes, ct_axes )

    def get_all_the_ways_to_get( self, axis_names, attributes, tensor_names, tensor_axes, matrix, vector ):
        for name, argument in self.sub_dict.items():
            argument.get_all_the_ways_to_get( axis_names, attributes + [ name ], tensor_names, tensor_axes, matrix, vector )

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        for argument in self.sub_dict.values():
            argument.get_arg_decl( non_differentiable_inputs, differentiable_inputs, parameters, outputs )

