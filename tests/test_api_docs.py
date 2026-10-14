"""The API reference of the site follows the code: `docs/api/*.md` is what `scripts/api_docs.py` writes from the docstrings
today, and every name that `sdot` exports is in a page ( or explicitly left out )."""

from errand import test


if test( "the_api_reference_is_up_to_date" ):
    import runpy
    from pathlib import Path
    script = runpy.run_path( str( Path( __file__ ).resolve().parents[ 1 ] / "scripts" / "api_docs.py" ) )
    script[ "main" ]( check = True )
