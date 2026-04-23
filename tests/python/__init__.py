from devtools import debug, sformat

def custom_printer_impl( obj, nl ):
    if isinstance( obj, ( list, tuple ) ):
        if len( obj ) == 0:
            if isinstance( obj, tuple ):
               return "()"
            return "[]"
        nnl = nl + "  "
        res = ""
        for v in obj:
            res += nnl + custom_printer_impl( v, nnl )
        return res

    if hasattr( obj, "__dict__" ):
        res = type( obj ).__name__
        nnl = nl + "  "
        for n, v in obj.__dict__.items():
            res += nnl + n + ": " + custom_printer_impl( v, nnl )
        return str(res)

    return repr( obj )

def custom_printer( *arg, **kwargs ):
    f = debug._process( arg, kwargs, frame_depth = 2 )
    s = sformat( f"{ f.filename }:{ f.lineno }:", sformat.blue )
    for argument in f.arguments:
        if s:
            s += "\n  "
        else:
            s += "  "
        s += sformat( argument.name, sformat.green ) + ": " + custom_printer_impl( argument.value, "\n  " )
    print( s )

builtins = __import__('builtins')
setattr( builtins, "info", custom_printer )
