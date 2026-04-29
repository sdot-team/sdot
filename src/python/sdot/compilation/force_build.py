import os

def force_build() -> bool:
    return "SDOT_FORCE_BUILD" in os.environ and bool( int( os.environ[ "SDOT_FORCE_BUILD" ] ) )
