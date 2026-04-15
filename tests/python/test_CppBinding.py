from sdot import cpp_binding, CtInt
import numpy

# import faulthandler                                                                                                                                                                                                                                                                                                    │
# faulthandler.enable()

if __name__ == "__main__":
    ic( cpp_binding( "test", "sdot/testouille.h" )( CtInt( 10 ) ) )
