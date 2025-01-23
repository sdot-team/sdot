from sdot.bindings.construct import construct, generate

def bindings( fout, scalar_type, periodicity, nb_dims ):
    fout.write( '\n' )
    fout.write( '#include "../symbolic_array_bindings.cpp"\n' )


construct( Environment, VariantDir, Configure, ARGLIST, "symbolic_array", [ "scalar_type", "periodicity", "nb_dims" ], [
    generate( "symbolic_array_bindings", bindings ),
] )



