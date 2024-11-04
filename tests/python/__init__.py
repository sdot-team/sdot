from pathlib import Path
import sys

p = str( Path( __file__ ).parents[ 2 ] / "src" / "python" )
sys.path.append( p )
