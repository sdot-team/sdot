from sdot.info import info, infox

builtins = __import__( 'builtins' )
setattr( builtins, "infox", infox )
setattr( builtins, "info", info )

