"""The VIEWER of a scene store as a pinned bundle ( sdot ADR 0001 §2 ).

A store's `index.html` is a small BOOTSTRAP: it names one version of the viewer by its file and its SHA-256, and looks for it

    1. at `?viewer=<url>` when the page's URL gives one ( taken as it is: whoever typed it chose it );
    2. in `viewer/` inside the store ( `sviz pack` puts it there );
    3. at the published URLs ( `URLS`, the documentation's site ), checked against the hash.

then writes it in place of itself ( `document.write`: the page stays at the store's address, and reads the manifest and the
chunks beside it as before ). The bundle is the page of `store_page` with its title left as `__SVIZ_TITLE__`, which the
bootstrap fills.

    viewer/sdot-viz-<version>-<sha256[:12]>.html

`viewer = "auto"` ( the default of `write_store_page` ): a bundle not listed in `published.json` ( a checkout, a version
never released ) is copied into the store, since no URL serves it; a published one is only named. `"copy"` always
copies it ( a store to send ), `"link"` never.

`sviz bundle DIR --publish` writes the bundle into DIR ( `docs/public/viewer`, served by the documentation's site ) and
lists it in `published.json`; `sviz pack` makes a store work offline; `sviz serve` and its `?viewer=installed` serve the
installed viewer instead of the pinned one ( development ).

`file://` cannot read the files beside a page: a store stands on its own under ANY static server
( `python -m http.server` ), and `sviz export --html` makes one self-contained file.
"""
from __future__ import annotations

import functools
import hashlib
import html
import json
import re
from pathlib import Path

URLS = [ "https://sdot-team.github.io/sdot/viewer/{file}" ]
TITLE = "__SVIZ_TITLE__"
PUBLISHED = Path( __file__ ).with_name( "published.json" )


def _version() -> str:
    """sdot's version: a checkout's own ( its `pyproject.toml` ), else the installed distribution's"""
    pyproject = Path( __file__ ).resolve().parents[ 3 ] / "pyproject.toml"
    if pyproject.is_file():
        m = re.search( r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M )
        if m:
            return m.group( 1 )
    try:
        from importlib.metadata import version
        return version( "sdot" )
    except Exception:
        return "dev"


@functools.lru_cache( maxsize = 1 )
def viewer_bytes() -> bytes:
    """the bundle of the INSTALLED viewer: the store page with its title to fill"""
    from .Visualizer import store_page
    return store_page( TITLE ).encode()


def viewer_id( data: bytes | None = None ) -> dict:
    """`{ version, sha256, file }` of a bundle ( the installed one by default )"""
    data = viewer_bytes() if data is None else data
    sha = hashlib.sha256( data ).hexdigest()
    return dict( version = _version(), sha256 = sha, file = f"sdot-viz-{ _version() }-{ sha[ :12 ] }.html" )


def published() -> set:
    """the hashes of the bundles the documentation's site serves"""
    try:
        return { e[ "sha256" ] for e in json.loads( PUBLISHED.read_text() ) }
    except ( OSError, ValueError ):
        return set()


def write_viewer( store, data: bytes | None = None, ident: dict | None = None ) -> Path:
    """the bundle in `store/viewer/`, unless it is there already"""
    data = viewer_bytes() if data is None else data
    ident = ident or viewer_id( data )
    f = Path( store ) / "viewer" / ident[ "file" ]
    if not f.exists():
        f.parent.mkdir( parents = True, exist_ok = True )
        tmp = f.with_name( f".{ f.name }.tmp" )
        tmp.write_bytes( data )
        tmp.replace( f )
    return f


def bootstrap_page( title: str, ident: dict | None = None, urls = None ) -> str:
    """the `index.html` of a store: finds the bundle `ident` names, checks it, and becomes it"""
    ident = ident or viewer_id()
    pin = dict( ident, urls = list( URLS if urls is None else urls ), title = title )
    pin_js = json.dumps( pin, separators = ( ",", ":" ) ).replace( "</", "<\\/" )
    return _BOOT.replace( "@TITLE@", html.escape( title ) ).replace( "@PIN@", pin_js )


def write_store_page( store, title: str, viewer: str = "auto" ):
    """`index.html` ( the bootstrap ) in `store`, and the bundle beside it as `viewer` says ( see the module )"""
    if viewer not in ( "auto", "copy", "link" ):
        raise ValueError( f"viewer = 'auto', 'copy' or 'link', not { viewer!r }" )
    ident = viewer_id()
    if viewer == "copy" or ( viewer == "auto" and ident[ "sha256" ] not in published() ):
        write_viewer( store, ident = ident )
    ( Path( store ) / "index.html" ).write_text( bootstrap_page( title, ident ) )


def pinned( store ) -> dict | None:
    """what the bootstrap of `store` names ( None: an older store, whose `index.html` is the whole viewer )"""
    try:
        text = ( Path( store ) / "index.html" ).read_text()
    except OSError:
        return None
    m = re.search( r'<script type="application/json" id="sdot-viz">(.*?)</script>', text, re.S )
    return json.loads( m.group( 1 ).replace( "<\\/", "</" ) ) if m else None


def _download( pin: dict ) -> bytes | None:
    import urllib.request
    for u in pin.get( "urls", URLS ):
        try:
            with urllib.request.urlopen( u.replace( "{file}", pin[ "file" ] ), timeout = 30 ) as r:
                data = r.read()
        except OSError:
            continue
        if hashlib.sha256( data ).hexdigest() == pin[ "sha256" ]:
            return data
    return None


def pack( store, upgrade: bool = False ) -> Path:
    """the viewer the store names, copied into `store/viewer/`: the store then works offline. From the installed viewer
    when it is that one, else downloaded and checked. `upgrade`: the store moved to the installed viewer first."""
    store = Path( store )
    pin = pinned( store )
    if pin is None or upgrade:
        try:
            title = json.loads( ( store / "manifest.json" ).read_text() ).get( "title", store.stem )
        except ( OSError, ValueError ):
            title = pin[ "title" ] if pin else store.stem
        write_store_page( store, title, viewer = "copy" )
        return store / "viewer" / viewer_id()[ "file" ]
    f = store / "viewer" / pin[ "file" ]
    if f.exists() and hashlib.sha256( f.read_bytes() ).hexdigest() == pin[ "sha256" ]:
        return f
    if pin[ "sha256" ] == viewer_id()[ "sha256" ]:
        return write_viewer( store, ident = pin )
    data = _download( pin )
    if data is None:
        raise RuntimeError( f"{ store }: viewer { pin[ 'file' ] } neither installed nor found at { pin.get( 'urls' ) }; "
                            f"`sviz pack --upgrade { store }` moves it to the installed viewer" )
    return write_viewer( store, data, pin )


def bundle( out_dir, publish: bool = False ) -> Path:
    """the installed bundle written into `out_dir` ( the documentation's `public/viewer` ); `publish`: listed in
    `published.json`, so that the stores written from now on only name it"""
    ident = viewer_id()
    f = Path( out_dir ) / ident[ "file" ]
    f.parent.mkdir( parents = True, exist_ok = True )
    f.write_bytes( viewer_bytes() )
    if publish and ident[ "sha256" ] not in published():
        try:
            listed = json.loads( PUBLISHED.read_text() )
        except ( OSError, ValueError ):
            listed = [ ]
        listed.append( ident )
        PUBLISHED.write_text( json.dumps( listed, indent = 1 ) + "\n" )
    return f


# the bootstrap: plain JS, no placeholder of the viewer's template ( `@...@` are filled by `bootstrap_page` )
_BOOT = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@TITLE@</title>
<script type="application/json" id="sdot-viz">@PIN@</script>
<style>
  body { font-family: sans-serif; margin: 0; padding: 24px; color: #555; background: #fff; }
  @media (prefers-color-scheme: dark) { body { color: #bbb; background: #1e1e1e; } }
  code { background: rgba(127,127,127,0.15); padding: 1px 4px; border-radius: 3px; }
</style>
</head>
<body>
<p id="boot">Loading the viewer…</p>
<script>
(async () => {
  const pin = JSON.parse(document.getElementById("sdot-viz").textContent);
  const say = h => { document.getElementById("boot").innerHTML = h; };
  const hex = async buf => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buf)), b => b.toString(16).padStart(2, "0")).join("");
  const asked = new URLSearchParams(location.search).get("viewer");
  const tries = [];
  if (asked && asked !== "installed") tries.push({ url: asked, check: false });
  tries.push({ url: "viewer/" + pin.file, check: "local" });
  for (const u of pin.urls) tries.push({ url: u.replace("{file}", pin.file), check: true });
  for (const t of tries) {
    let buf;
    try {
      const r = await fetch(t.url, { cache: "no-cache" });
      if (!r.ok) continue;
      buf = await r.arrayBuffer();
    } catch (e) { continue; }
    if (t.check) {
      if (!(window.crypto && crypto.subtle)) {       // ( no digest outside a secure context: trust only the store's own copy )
        if (t.check !== "local") continue;
      } else if (await hex(buf) !== pin.sha256) {
        console.warn("sdot viewer: " + t.url + " is not " + pin.file);
        continue;
      }
    }
    const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
    const page = new TextDecoder().decode(buf).split("__SVIZ_TITLE__").join(esc(pin.title));
    document.open(); document.write(page); document.close();
    return;
  }
  if (location.protocol === "file:")
  {
    // ( a browser reads no file beside a page opened as a file: the store is opened through errand, which serves it )
    const dir = decodeURIComponent(location.pathname.replace(/\/[^/]*$/, ""));
    const cmd = "errand open '" + dir.replace(/'/g, "'\\''") + "'";
    say("A store opens through a server, not as a file. In a terminal:<br><code id=\"cmd\" style=\"user-select: all\"></code>" +
        "<br><small>( it serves the store beside <code>errand web</code>, with the services it asks for, and opens it here )</small>");
    document.getElementById("cmd").textContent = cmd;
  }
  else
    say("Viewer <code>" + pin.file + "</code> not found: neither in <code>viewer/</code> nor at " + pin.urls.map(u => u.replace("{file}", pin.file)).join(", ") +
        ".<br>Copy it here with <code>sviz pack DIR</code> ( <code>--upgrade</code> for the installed viewer ).");
})();
</script>
</body>
</html>
"""
