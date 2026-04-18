"""sdot.generated_files — storage for compiled extensions and generated C++ sources.

"""

# Make ``sdot.generated_files.<name>`` importable transparently.
from .compilation_directories import dylib_dir
__path__.append( str( dylib_dir() ) )
