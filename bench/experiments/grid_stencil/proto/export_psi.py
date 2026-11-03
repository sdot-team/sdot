"""Write final + intermediate psi of a few OT solves ( N=32 ) with the true ring of every cell."""
import sys
import numpy as np
sys.path.insert( 0, __file__.rsplit( "/", 1 )[ 0 ] )
import ot_stats as O
cases = []
for kind, c in [ ( "bands", 10 ), ( "bands", 100 ), ( "bands", 1000 ), ( "noise", 1000 ) ]:
    its = O.newton( O.masses( kind, c ) )
    for t, ( psi, ar, rings ) in enumerate( its ):
        if t in ( 1, len( its ) // 2, len( its ) - 1 ):
            cases.append( ( f"{kind}{c}_it{t}", psi, rings ) )
with open( "bench/psi_cases.bin", "wb" ) as f:
    np.array( [ len( cases ), O.N ], np.int32 ).tofile( f )
    for name, psi, rings in cases:
        np.frombuffer( name.ljust( 32 ).encode(), np.uint8 ).tofile( f )
        psi.astype( np.float64 ).tofile( f )
        rings.astype( np.int32 ).tofile( f )
print( len( cases ), "cases" )
