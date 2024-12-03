from sdot.bindings.construct import construct

construct( Environment, VariantDir, Configure, ARGLIST, "sdot", [ 'scalar_type', 'nb_dims' ], [
    'build/sdot_bindings.cpp',
], add_srcs_for_windows = [
    "ext/tl20/src/cpp/tl/support/display/DisplayItem_Pointer.cpp",
    "ext/tl20/src/cpp/tl/support/display/DisplayItem_Number.cpp",
    "ext/tl20/src/cpp/tl/support/display/DisplayItem_String.cpp",
    "ext/tl20/src/cpp/tl/support/display/DisplayItem_List.cpp",

    "ext/tl20/src/cpp/tl/support/display/DisplayParameters.cpp",
    "ext/tl20/src/cpp/tl/support/display/DisplayContext.cpp",
    "ext/tl20/src/cpp/tl/support/display/DisplayItem.cpp",

    "ext/tl20/src/cpp/tl/support/string/CompactReprWriter.cpp",
    "ext/tl20/src/cpp/tl/support/string/CompactReprReader.cpp",
    "ext/tl20/src/cpp/tl/support/string/read_arg_name.cpp",
    "ext/tl20/src/cpp/tl/support/string/va_string.cpp",
    
    "ext/tl20/src/cpp/tl/support/Displayer.cpp",

    'build/cpp/sdot/support/BigRational.cpp',
    "build/cpp/sdot/support/VtkOutput.cpp",
    "build/cpp/sdot/support/Mpi.cpp",

    'build/cpp/sdot/symbolic/instructions/Symbol.cpp',
    'build/cpp/sdot/symbolic/instructions/Value.cpp',
    'build/cpp/sdot/symbolic/instructions/Func.cpp',
    'build/cpp/sdot/symbolic/instructions/Inst.cpp',
    'build/cpp/sdot/symbolic/Expr.cpp',
] )

