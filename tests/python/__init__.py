def info( *args, **kwargs ):
    from sdot.util.info import info as _info
    return _info( *args, frame_depth = 3, **kwargs )

def infox( *args, **kwargs ):
    from sdot.util.info import infox as _infox
    return _infox( *args, frame_depth = 3, **kwargs )

builtins = __import__( 'builtins' )
setattr( builtins, "info", info )
setattr( builtins, "infox", infox )

