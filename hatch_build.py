import sys
import sysconfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from pathlib import Path


class CustomBuildHook( BuildHookInterface ):
    def initialize( self, version, build_data ):
        generated_dir = Path( self.root ) / 'src' / 'python' / 'sdot' / 'generated_files'
        extensions    = ( '*.so', '*.dll', '*.pyd', '*.dylib' )

        dylibs = [ f for pattern in extensions for f in generated_dir.glob( pattern ) ]
        if not dylibs:
            return

        # Tag explicite : cp313-cp313-macosx_15_0_arm64 (ou équivalent plateforme)
        # L'inclusion des .so est gérée par 'artifacts' dans pyproject.toml
        vi           = sys.version_info
        python_tag   = f'cp{ vi.major }{ vi.minor }'
        platform_tag = sysconfig.get_platform().replace( '-', '_' ).replace( '.', '_' )
        build_data[ 'tag' ] = f'{ python_tag }-{ python_tag }-{ platform_tag }'
