"""The code of the documentation runs: every file of `docs/snippets/` is included as it is in a page ( `<<< @/snippets/...` ),
so a snippet that breaks is a page that lies. Each one runs in its own process and directory ( some write a page ), and a snippet
written for a framework that the environment does not have skips itself."""

from errand import test, skip


def _run( name ):
    """runs the snippet in a process of its own: a snippet that imports torch must not change what the next one builds"""
    import importlib.util
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    for framework in ( "jax", "torch", "cupy" ):
        if framework in name and importlib.util.find_spec( framework ) is None:
            skip( f"{ framework } is not installed" )
    path = Path( __file__ ).resolve().parents[ 1 ] / "docs" / "snippets" / name
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run( [ sys.executable, str( path ) ], cwd = tmp, capture_output = True, text = True )
    assert res.returncode == 0, f"{ name } failed:\n{ res.stdout[ -2000: ] }{ res.stderr[ -4000: ] }"


if test( "hello_numpy" ):
    _run( "hello_numpy.py" )

if test( "hello_jax" ):
    _run( "hello_jax.py" )

if test( "hello_torch" ):
    _run( "hello_torch.py" )

if test( "hello_cupy" ):
    _run( "hello_cupy.py" )

if test( "plan" ):
    _run( "plan.py" )

if test( "cell_complex" ):
    _run( "cell_complex.py" )

if test( "ot_solve" ):
    _run( "ot_solve.py" )

if test( "ot_problem" ):
    _run( "ot_problem.py" )

if test( "distributions" ):
    _run( "distributions.py" )

if test( "distributions_1d" ):
    _run( "distributions_1d.py" )

if test( "grad_jax" ):
    _run( "grad_jax.py" )

if test( "grad_torch" ):
    _run( "grad_torch.py" )

if test( "grad_formula" ):
    _run( "grad_formula.py" )

if test( "duplicates_diagram" ):
    _run( "duplicates_diagram.py" )

if test( "duplicates_plan" ):
    _run( "duplicates_plan.py" )


def _run_launched( name, launcher, module = None ):
    """runs the snippet the way its page says: under `torchrun` or `mpiexec`, 4 processes"""
    import importlib.util
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    if module is not None and importlib.util.find_spec( module ) is None:
        skip( f"{ module } is not installed" )
    path = Path( __file__ ).resolve().parents[ 1 ] / "docs" / "snippets" / name
    if launcher == "torchrun":
        cmd = [ sys.executable, "-m", "torch.distributed.run", "--nproc-per-node", "4", str( path ) ]
    else:
        mpiexec = shutil.which( "mpiexec", path = os.path.dirname( sys.executable ) ) or shutil.which( "mpiexec" )
        cmd = [ mpiexec, "-n", "4", sys.executable, str( path ) ]
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run( cmd, cwd = tmp, capture_output = True, text = True, timeout = 900 )
    assert res.returncode == 0, f"{ name } failed:\n{ res.stdout[ -2000: ] }{ res.stderr[ -4000: ] }"


if test( "parallel_boxes" ):
    _run( "parallel_boxes.py" )

if test( "parallel_jax" ):
    _run( "parallel_jax.py" )

if test( "parallel_program" ):
    # ( JAX ranks: `_run` only skips on the name )
    import importlib.util
    if importlib.util.find_spec( "jax" ) is None:
        skip( "jax is not installed" )
    _run( "parallel_program.py" )

if test( "parallel_torch" ):
    _run_launched( "parallel_torch.py", "torchrun", "torch" )

if test( "parallel_mpi" ):
    _run_launched( "parallel_mpi.py", "mpiexec", "mpi4py" )
