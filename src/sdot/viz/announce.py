"""A page written, and where it opens: `OUTPUT: file://...`, a link a terminal opens with a ctrl+click.

The page is written beside its place then renamed over it: a reader never sees half of it, not even
errand bringing it back while the run writes it again. Under errand, `errand.view` says where it opens:
a run on another machine has its file brought back at once and said by its path on the machine that
drives it, where the link opens."""
import os
from pathlib import Path


def write_page( path, text: str ) -> Path:
    path = Path( path )
    path.parent.mkdir( parents = True, exist_ok = True )
    tmp = path.with_name( f".{ path.name }.{ os.getpid() }.tmp" )
    tmp.write_text( text )
    tmp.replace( path )
    announce( path )
    return path


def announce( path ):
    path = Path( path )
    try:
        import errand
        if errand.view( path.absolute() ):
            return
    except ImportError:
        pass
    print( f"OUTPUT: { path.absolute().as_uri() }" )
