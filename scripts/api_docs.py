"""The API reference of the site ( `docs/api/*.md` ), written from the docstrings -- so that it follows the code.

    python scripts/api_docs.py            # writes the pages
    python scripts/api_docs.py --check    # fails if a page is not what the code says ( `tests/test_api_docs.py` )

What goes where is said in `PAGES`; a name that `sdot` exports must be in a page or in `EXCLUDED`, otherwise this
fails: a new export is never silently left out of the reference.

What is read, for each name:

  * a function: its signature and its docstring;
  * a class: the signature of its constructor, its OWN docstring ( not the one it inherits from loom's `Aggregate` ) or,
    when it has none or says "see the module docstring", that of the module of the same name; then its attributes
    documented by a `#:` comment ( `#: the transport cost` above `self.cost = ...` ), its properties and its methods --
    those of the classes of `sdot` it derives from included.

A docstring is written for the user first. Notes for the developers follow a line `= Internals`, and are not shown;
a member whose docstring has no user part ( none at all, or `= Internals` from its first line ) is not shown either.
"""
import inspect
import re
import sys
from pathlib import Path

ROOT = Path( __file__ ).resolve().parents[ 1 ]
OUT = ROOT / "docs" / "api"

PAGES = {
    "ot-plan": ( "W2 and Plans", """The transport: `w2` for the distance, `plan` for the plan, `ot_solve` for the plan with the settings of
the solver, `OtProblem` to solve again and again. See [Guide: plans and settings](/guide/ot-plan).""",
                 [ "w2", "plan", "ot_solve", "OtProblem", "Iterative", "Direct", "OtNotConverged", "SdotPlanNd", "SdotPlan1d",
                   "quantization", "wrap" ] ),
    "distributions": ( "Distributions", """The sources ( diracs ) and the targets ( densities ). See [Guide: Distributions](/guide/distributions/).""",
                       [ "SumOfDiracs", "SumOfDiracs1d", "ProjectedSumOfDiracs", "Box", "ConvexHull", "Polytope",
                         "Image", "SplineGrid", "SumOfGaussians", "Mesh", "CodimMesh" ] ),
    "power-diagram": ( "Power Diagrams", """The geometry under the plans: power diagrams, their cells, and what speeds them up. For seeds at the same position, see [Guide: Duplicates and Close Seeds](/guide/duplicates).""",
                       [ "PowerDiagram", "Groups", "Voronoi", "Cell", "CellComplex", "Faces", "Face", "CellGeometry", "box_half_spaces", "AaBsp", "SpatialAccelerator", "set_kernel_dtype",
                         "kernel_dtype" ] ),
    "visualization": ( "Visualization", """Pages to look at what was computed ( `show`, `write_html` of the objects use them ).""",
                       [ "Visualizer", "write_convergence_html" ] ),
}

# exported, but not for the reference: the bench settings ( unstable ), and the variants a constructor picks by itself
EXCLUDED = { "Tuning", "CellScratch", "Cell_1", "Cell_2", "Cell_N", "PowerDiagram_Bsp", "PowerDiagram_Plain" }


def internal( value ):
    """the function or the class behind a public name ( `loom.export_namespace` wraps them )"""
    if isinstance( value, type ):
        return value.__dict__.get( "_loom_cls", value )
    return getattr( value, "__wrapped__", value )


def user_part( doc ):
    """what is before the line `= Internals`"""
    lines = ( doc or "" ).splitlines()
    for k, line in enumerate( lines ):
        if line.strip() == "= Internals":
            lines = lines[ :k ]
            break
    return "\n".join( lines ).strip()


def own_doc( obj ):
    """the docstring of `obj` itself; for a class without one ( or that sends to its module ), that of its module"""
    doc = obj.__dict__.get( "__doc__" ) if isinstance( obj, type ) else obj.__doc__
    doc = inspect.cleandoc( doc ) if doc and doc.strip() else ""
    if isinstance( obj, type ) and ( not doc or doc.lower().startswith( "see the module" ) ):
        module = sys.modules.get( obj.__module__ )
        if module is not None and module.__doc__ and module.__name__.rsplit( ".", 1 )[ -1 ] == obj.__name__:
            doc = inspect.cleandoc( module.__doc__ )
    return user_part( doc )


def to_markdown( doc ):
    """a docstring as markdown: `= Title` lines become titles, `<` and `{{` are escaped outside the code"""
    out = []
    for line in doc.splitlines():
        m = re.match( r"^= (.*)$", line )
        if m:
            out.append( f"**{ m.group( 1 ).strip() }**" )
            continue
        if line.startswith( "    " ):
            out.append( line )               # an indented block: code, as it is
            continue
        parts = re.split( r"(`[^`]*`)", line )
        out.append( "".join( p if p.startswith( "`" ) else p.replace( "<", "&lt;" ) for p in parts ) )
    return "\n".join( out )


def signature( obj, name ):
    try:
        sig = str( inspect.signature( obj ) )
    except ( TypeError, ValueError ):
        sig = "( ... )"
    sig = re.sub( r"\s*->.*$", "", sig )            # the internal types: the user gets arrays and views
    sig = sig.replace( "(self, ", "(" ).replace( "(self)", "()" ).replace( "(cls, ", "(" ).replace( "(cls)", "()" )
    return f"{ name }{ sig }"


def sdot_classes( cls ):
    return [ c for c in cls.__mro__ if c.__module__.startswith( "sdot" ) ]


def documented_attributes( cls ):
    """`{ name: text }`: the `self.name = ...` that a `#:` comment introduces, in the classes of `sdot` of `cls`"""
    res = {}
    for c in reversed( sdot_classes( cls ) ):
        try:
            src = inspect.getsource( c )
        except ( OSError, TypeError ):
            continue
        comment, closed = [], False      # a `#:` comment documents the next `self.x =` of the same method
        for line in src.splitlines():
            s = line.strip()
            if s.startswith( "#:" ):
                if closed:
                    comment, closed = [], False
                comment.append( s[ 2: ].strip() )
                continue
            closed = closed or bool( comment )
            if s.startswith( ( "def ", "class ", "@" ) ):
                comment, closed = [], False
                continue
            m = re.match( r"self\.([a-zA-Z]\w*)\s*(:[^=]*)?=", s )
            if m and comment:
                if m.group( 1 ) not in res:
                    res[ m.group( 1 ) ] = " ".join( comment )
                comment, closed = [], False
    return res


def members( cls ):
    """`( properties, methods )`: `[ ( name, object ) ]` defined in the classes of `sdot` of `cls`, the nearest first"""
    props, methods, seen = [], [], set()
    for c in sdot_classes( cls ):
        for name, value in c.__dict__.items():
            if name.startswith( "_" ) or name in seen:
                continue
            seen.add( name )
            if isinstance( value, property ):
                if user_part( inspect.getdoc( value ) ):
                    props.append( ( name, value ) )
            elif isinstance( value, ( staticmethod, classmethod ) ):
                if user_part( inspect.getdoc( value.__func__ ) ):
                    methods.append( ( name, value.__func__ ) )
            elif inspect.isfunction( value ):
                if user_part( inspect.getdoc( value ) ):
                    methods.append( ( name, value ) )
    return sorted( props ), sorted( methods )


def entry( name, value ):
    obj = internal( value )
    lines = [ f"## `{ name }`", "" ]
    if isinstance( obj, type ):
        lines += [ "```python", signature( obj, name ), "```", "" ]
    else:
        lines += [ "```python", signature( obj, name ), "```", "" ]
    doc = own_doc( obj )
    if doc:
        lines += [ to_markdown( doc ), "" ]
    if not isinstance( obj, type ) or issubclass( obj, BaseException ):
        return lines

    attributes = documented_attributes( obj )
    props, methods = members( obj )
    rows = [ ( n, t ) for n, t in attributes.items() ] + \
           [ ( n, user_part( inspect.getdoc( p ) ).split( "\n\n" )[ 0 ].replace( "\n", " " ) ) for n, p in props if n not in attributes ]
    if rows:
        lines += [ "| Attribute | |", "|---|---|" ]
        lines += [ f"| `{ n }` | { to_markdown( t ).replace( '|', '\\|' ) } |" for n, t in sorted( rows ) ]
        lines.append( "" )
    for n, fn in methods:
        lines += [ f"### `.{ signature( fn, n ) }`", "" ]
        lines += [ to_markdown( user_part( inspect.getdoc( fn ) ) ), "" ]
    return lines


def pages():
    import sdot
    exported = { n for n in dir( sdot ) if not n.startswith( "_" ) and not inspect.ismodule( getattr( sdot, n ) ) }
    listed = { n for _, _, names in PAGES.values() for n in names }
    missing = exported - listed - EXCLUDED
    if missing:
        raise RuntimeError( f"api_docs: exported by sdot but in no page ( nor in EXCLUDED ): { sorted( missing ) }" )
    res = {}
    for stem, ( title, intro, names ) in PAGES.items():
        lines = [ f"<!-- written by scripts/api_docs.py from the docstrings: do not edit -->", "", f"# API — { title }", "",
                  intro, "", "::: v-pre", "" ]
        for n in names:
            lines += entry( n, getattr( sdot, n ) )
        lines += [ ":::", "" ]
        res[ OUT / f"{ stem }.md" ] = "\n".join( lines )
    return res


def main( check = False ):
    stale = []
    for path, text in pages().items():
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append( path.name )
        else:
            path.write_text( text )
    if stale:
        raise RuntimeError( f"api_docs: not up to date with the docstrings: { stale } -- run `python scripts/api_docs.py`" )


if __name__ == "__main__":
    try:
        main( check = "--check" in sys.argv )
    except RuntimeError as e:
        raise SystemExit( str( e ) )
