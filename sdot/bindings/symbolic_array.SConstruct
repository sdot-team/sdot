from sdot.bindings.construct import construct, generate

# def bindings( fout, scalar_type, periodicity, nb_dims ):
#     fout.write( f'namespace sdot {{\n' )
#     fout.write( f'struct IndHandling_{ periodicity } {{\n' )
#     fout.write( f'    static PI index( ) {{\n' )
#     fout.write( f'    }}\n' )
#     fout.write( f'}};\n' )
#     fout.write( f'}} // namespace sdot\n' )
#     fout.write( f'#define SYMBOLIC_ARRAY_IND_HANDLING IndHandling_{ periodicity }\n' )
#     fout.write( '#include "../symbolic_array_bindings.cpp"\n' )


construct( Environment, VariantDir, Configure, ARGLIST, "symbolic_array", [ "scalar_type", "nb_dims" ], [
    # generate( "symbolic_array_bindings", bindings ),
    "build/symbolic_array_bindings.cpp",
] )



