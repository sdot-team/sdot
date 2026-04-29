from sdot.util.info import info, infox

builtins = __import__( 'builtins' )
setattr( builtins, "infox", infox )
setattr( builtins, "info", info )

