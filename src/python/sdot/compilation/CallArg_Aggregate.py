from ..util.get_all_annotations import get_all_annotations
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

        template_args : dict[ str ] = {}
        for argument in self.sub_dict.values():
            argument.get_template_args( template_args )
            argument.get_includes( includes )

        lines.append( f"template<{ str.join( ",", [ f"{ t } { n }" for n, t in template_args.items() ] ) }>" )
        lines.append( f"struct { self.base_cpp_name() } {{" )
        for name, argument in self.sub_dict.items():
            lines.append( f"    { argument.cpp_type_name( self ) } { name };" )
        lines.append( "};" )

        lines += [ "", "} // namespace sdot", "" ]

        code = "\n".join(
            [ f"#include <{ include }>" for include in includes ] + \
            lines
        )

        infox( code )

        #
        from ..generated_files.compilation_directories import generated_includes_dir
        path = generated_includes_dir() / f"{ self.base_cpp_name() }.h"
        path.write_text( code )



    def generate_structures( self ):
        self.generate_structure()

        for argument in self.arguments.values():
            argument.generate_structures()
