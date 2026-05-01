from ..util.get_all_annotations import get_all_annotations

from .SubDictContainer import SubDictContainer
from .IoCategory import IoCategory
from .CallArg import CallArg

class CallArg_Aggregate( SubDictContainer, CallArg ):
    """ input or mutable
    """

    python_class: any #
    python_value: any #
    io_category : IoCategory #
    sub_dict    : dict[ CallArg ] #
    # ct_axes     : dict

    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        res = CallArg_Aggregate()

        res.python_class = python_class
        res.python_value = python_value
        res.io_category = io_category
        res.sub_dict = {}
        # res.ct_axes = {}

        for name, annotation in get_all_annotations( python_class ).items():
            value = None
            if python_value is not None:
                value = getattr( python_value, name )
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
        template_args = {}
        self.get_template_args( template_args, names )

        if self.python_value:
            res = self.python_value.__class__.__name__
        else:
            res = self.python_class.__name__

        if len( template_args ):
            res += f"<{ ','.join( template_args ) }>"

        return res

    def base_cpp_name( self ) -> str:
        return self.python_class.__name__

    def assemble_return( self ):
        ctor_args = {}
        for name, call_arg in self.sub_dict.items():
            ctor_args[ name ] = call_arg.assemble_return()
        return self.python_class( **ctor_args )

    def get_includes( self, includes: set ):
        includes.add( f"sdot/generated_includes/{ self.base_cpp_name() }.h" )

    def generate_structure( self ):
        includes = set( [ "sdot/support/DynamicAxis.h" ] )
        beg_lines = [ "", "namespace sdot {", "" ]
        end_lines = [ "", "} // namespace sdot", "" ]

        code = self.struct_decl( self.base_cpp_name(), includes, beg_lines, end_lines )

        from ..generated_files.compilation_directories import generated_includes_dir
        path = generated_includes_dir() / f"{ self.base_cpp_name() }.h"
        try:
            old_text = path.read_text()
        except FileNotFoundError:
            old_text = ""
        if code != old_text:
            path.write_text( code )

    def generate_structures( self ):
        self.generate_structure()

        for argument in self.sub_dict.values():
            argument.generate_structures()

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        pass

    def get_all_the_ways_to_get( self, axis_names, attributes, use_attributes, tensor_names, tensor_axes, matrix, vector ):
        for name, argument in self.sub_dict.items():
            argument.get_all_the_ways_to_get( axis_names, attributes + [ name ], use_attributes, tensor_names, tensor_axes, matrix, vector )

    def get_arg_decl( self, non_differentiable_inputs: list, differentiable_inputs: list, parameters: list, outputs: list ):
        for argument in self.sub_dict.values():
            argument.get_arg_decl( non_differentiable_inputs, differentiable_inputs, parameters, outputs )

    def assembled_code( self, beg_line ):
        return super().assembled_code( self.base_cpp_name(), beg_line )
