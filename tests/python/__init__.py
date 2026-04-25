from sdot.info import info

builtins = __import__( 'builtins' )
setattr( builtins, "info", info )

