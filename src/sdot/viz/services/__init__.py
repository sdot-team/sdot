"""The services a store's page can ask for ( sdot ADR 0001 §3, errand ADR 0026 / 0027 ): HTTP servers that know their store
from `X-Errand-Root` ( relayed by errand ) or from `--root` ( started by hand ), and answer in the chunk format of `store.py`.

    sdot.ao     `python -m sdot.viz.services.ao`: the baked occlusion of a frame ( `occlusion.py` )
    blender     `blender -b -P .../services/blender.py`: a picture of the current view by Cycles ( Blender's Python )
    sdot.region `python -m sdot.viz.services.region`: the cells of a region, rebuilt from the frame's sources

How each one runs is DATA, in `errand-services.json` beside this file: the words of its command ( `{python}`, `{file:…}`,
`{tool:blender}`, `{port}`, resolved by errand on the machine that runs it ), the files that make its version, its needs,
and the tools it uses ( Blender: where to look, where to read its version, the pinned downloads ). errand finds that file
through the entry point `errand.services` of the installed package, without importing sdot.

`declare( store )` writes the store's `services.json` ( format 2 ): for each service, its name, its version ( `version`,
the algorithm of errand ADR 0027 §2 ), the package that provides it, and a copy of its description. No path, no Python of
the writer's: the same store gets lines tailored to whichever machine it is opened on.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path( __file__ ).resolve().parent
DESCRIBED = HERE / "errand-services.json"


def described( ) -> dict:
    """this package's `errand-services.json`"""
    return json.loads( DESCRIBED.read_text() )


SERVICES = described()[ "services" ]


def version( name: str, entry: dict | None = None, base = HERE ) -> str:
    """`name#version`'s version: 8 hex digits of a sha256 over the protocol and the files `version_of` lists, in order --
    the same as errand computes on each machine ( errand ADR 0027 §2 )"""
    entry = entry if entry is not None else SERVICES[ name ]
    h = hashlib.sha256( f"errand-service/1 {name} protocol {entry.get( 'protocol', 1 )}\n".encode() )
    for rel in entry.get( "version_of" ) or [ ]:
        data = ( Path( base ) / rel ).read_bytes()
        h.update( f"{rel} {len( data )}\n".encode() )
        h.update( data )
    return h.hexdigest()[ :8 ]


def _package_version( ) -> str | None:
    try:
        from importlib.metadata import version as dist_version
        return dist_version( "sdot" )
    except Exception:
        return None


def declare( store, dims: int | None = None ):
    """`services.json` in `store`: the services that make sense for it ( all three are about 3D scenes )"""
    d = described()
    pkg = { "name": d[ "package" ][ "name" ], "version": _package_version() }
    services = { }
    for name, entry in d[ "services" ].items():
        if dims not in ( None, 3 ):
            continue
        services[ name ] = { "version": version( name, entry ), "protocol": entry.get( "protocol", 1 ), "optional": True,
                             "text": entry.get( "text", "" ), "package": pkg,
                             "described": { **entry, "package": d[ "package" ] } }
    path = Path( store ) / "services.json"
    tmp = path.with_name( f".services.json.{ os.getpid() }.tmp" )
    tmp.write_text( json.dumps( { "format": 2, "services": services }, indent = 1 ) + "\n" )
    os.replace( tmp, path )
