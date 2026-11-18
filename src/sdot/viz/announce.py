"""Where a written page opens: `OUTPUT: file://...`, a link a terminal opens with a ctrl+click.

Under errand, `errand.view` says it instead: a run on another machine has its file brought back at once
and said by its path on the machine that drives it, where the link opens."""
from pathlib import Path


def announce( path ):
    path = Path( path )
    try:
        import errand
        if errand.view( path.absolute() ):
            return
    except ImportError:
        pass
    print( f"OUTPUT: { path.absolute().as_uri() }" )
