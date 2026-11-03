# Checks `benchlib`'s clock sampler: one busy core, then every core busy ( numpy's BLAS ).
#   errand run --env lmo-numpy --no-queue python sdot-ffi/bench/experiments/clock_check.py
import os, sys, time
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
import numpy as np
import benchlib

def busy( seconds, f ):
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        f()

a = np.random.rand( 2000, 2000 )
benchlib.start_clock(); busy( 3, lambda: sum( range( 100000 ) ) ); print( "one core ", benchlib.hardware().get( "clock_ghz" ) )
benchlib.start_clock(); busy( 3, lambda: a @ a ); print( "all cores", benchlib.hardware().get( "clock_ghz" ) )
print( "nominal  ", benchlib.hardware()[ "hw_cpu" ] )
