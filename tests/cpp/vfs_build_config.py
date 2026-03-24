import os, sys

def config( options ):
    options.load_lib( 'https://github.com/catchorg/Catch2.git', lib_names = [ "Catch2Main", "Catch2" ] )

    options.add_inc_path( '../subprojects/zpp_bits' )
    options.add_inc_path( '../src/cpp' )
    # vfs.vfs_build_config( options )

    options.add_cpp_flag( '-std=c++20' )
    options.add_cpp_flag( '-fPIC' )
    options.add_cpp_flag( '-g3' )


    # $(llvm-config --cxxflags --ldflags --system-libs --libs core executionengine jitlinker native)
    #   export CMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm"
    # if sys.platform.startswith( "linux" ):
    #     options.add_inc_path( '/home/leclerc/Prog/llvm/LLVM-21/include' )
    #     options.add_lib_path( '/home/leclerc/Prog/llvm/LLVM-21/lib' )
    #     for l in str.split( "-lLLVMX86TargetMCA -lLLVMMCA -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info -lLLVMMCDisassembler -lLLVMAsmPrinter -lLLVMOrcJIT -lLLVMPasses -lLLVMIRPrinter -lLLVMHipStdPar -lLLVMCoroutines -lLLVMipo -lLLVMInstrumentation -lLLVMVectorize -lLLVMSandboxIR -lLLVMLinker -lLLVMFrontendOpenMP -lLLVMFrontendDirective -lLLVMFrontendAtomic -lLLVMFrontendOffloading -lLLVMObjectYAML -lLLVMGlobalISel -lLLVMSelectionDAG -lLLVMCodeGen -lLLVMScalarOpts -lLLVMInstCombine -lLLVMObjCARCOpts -lLLVMCodeGenTypes -lLLVMCGData -lLLVMBitWriter -lLLVMCFGuard -lLLVMAggressiveInstCombine -lLLVMTransformUtils -lLLVMWindowsDriver -lLLVMJITLink -lLLVMOption -lLLVMExecutionEngine -lLLVMTarget -lLLVMAnalysis -lLLVMProfileData -lLLVMSymbolize -lLLVMDebugInfoBTF -lLLVMDebugInfoPDB -lLLVMDebugInfoMSF -lLLVMDebugInfoCodeView -lLLVMDebugInfoGSYM -lLLVMDebugInfoDWARF -lLLVMFrontendHLSL -lLLVMRuntimeDyld -lLLVMOrcTargetProcess -lLLVMOrcShared -lLLVMObject -lLLVMTextAPI -lLLVMMCParser -lLLVMIRReader -lLLVMAsmParser -lLLVMBitReader -lLLVMMC -lLLVMDebugInfoDWARFLowLevel -lLLVMCore -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMTargetParser -lLLVMSupport -lLLVMDemangle" ):
    #         options.add_lib_name( l[ 2: ] )
    #     options.add_lib_name( "xml2" )
    #     options.add_lib_name( "rt" )
    #     options.add_lib_name( "dl" )
    #     options.add_lib_name( "m" )
    #     options.add_lib_name( "z" )
    #     # -lrt -ldl -lm -lz -lxml2
    # else:
    #     options.add_inc_path( '/opt/homebrew/opt/llvm/include' )
    #     options.add_lib_path( '/opt/homebrew/opt/llvm/lib' )
    #     options.add_lib_name( 'llvm' )

    # options.add_define( 'TL_INSTALL_DIR="' + os.getcwd() + '"' )

