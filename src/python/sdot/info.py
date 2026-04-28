from .drivers.compilation.collect_attributes import collect_attributes_inst
from devtools import debug, sformat
import sys

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

    if type( obj ).__str__ is not object.__str__:
        return obj.__str__()

    if hasattr( obj, "__dict__" ):
        res = type( obj ).__name__
        nnl = nl + "  "
        for n, _ in collect_attributes_inst( obj, use_annotations = True ):
            res += nnl + n + ": " + custom_printer_impl( getattr( obj, n, None ), nnl )
        return str(res)

    return str( obj )

def info( *arg, **kwargs ):
    f = debug._process( arg, kwargs, frame_depth = 2 )
    s = sformat( f"{ f.filename }:{ f.lineno }:", sformat.blue )
    for argument in f.arguments:
        if s:
            s += "\n  "
        else:
            s += "  "
        s += sformat( argument.name, sformat.green ) + ": " + custom_printer_impl( argument.value, "\n  " )
    print( s )

def infox( *arg, **kwargs ):
    f = debug._process( arg, kwargs, frame_depth = 2 )
    s = sformat( f"{ f.filename }:{ f.lineno }:", sformat.blue )
    for argument in f.arguments:
        if s:
            s += "\n  "
        else:
            s += "  "
        s += sformat( argument.name, sformat.green ) + ": " + custom_printer_impl( argument.value, "\n  " )
    print( s )
    sys.exit( 0 )
