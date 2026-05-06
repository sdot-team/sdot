"""Bump la version date-based dans pyproject.toml et crée le tag git correspondant.

Schéma : YYYY.MM.DD.N  (N commence à 1 et s'incrémente si plusieurs releases le même jour)
"""
from pathlib import Path
from datetime import date
import re
import subprocess
import sys


def main():
    today = date.today().strftime( '%Y.%m.%d' )
    path  = Path( __file__ ).parents[ 1 ] / 'pyproject.toml'
    text  = path.read_text()

    m = re.search( r'version = "(\d{4}\.\d{2}\.\d{2})\.(\d+)"', text )
    if m:
        cur_date, n = m.group( 1 ), int( m.group( 2 ) )
        new_n = n + 1 if cur_date == today else 1
    else:
        new_n = 1

    version = f'{ today }.{ new_n }'
    text    = re.sub( r'version = "\d{4}\.\d{2}\.\d{2}\.\d+"',
                      f'version = "{ version }"', text )
    path.write_text( text )

    tag = f'v{ version }'
    subprocess.run( [ 'git', 'add', str( path ) ],             check=True )
    subprocess.run( [ 'git', 'commit', '-m', f'release { version }' ], check=True )
    subprocess.run( [ 'git', 'tag', tag ],                     check=True )

    print( f'Version { version } — tag { tag } créé.' )
    print( f'Pour publier : git push && git push origin { tag }' )


main()
