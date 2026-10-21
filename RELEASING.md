# Releasing sdot

One command builds every sdot wheel and brings them into `dist/`:

```bash
make wheels      # = errand run --env mac_wheels,lin_wheels,win_wheels --pull dist make local_wheels
```

Each environment ( `errand-envs.py` ) runs `make local_wheels` on its own copy of the tree: the kernels of every framework
( `scripts/wheels.py --kernels` ), then its sdot wheel, validated in a clean virtual environment ( with a wheel of the loom
checkout, built on the side and not kept ). `--pull dist` brings the result back. The x86 levels a wheel carries are decided
by measure while its kernels are built ( `level_workloads` in `src/sdot/catalogue.py`, `levels.json` in the catalogue ): run
it on a quiet machine, or a level that gains a little is dropped as noise.
The hosts and paths of the machines are in `errand-vars.py`, which is not versioned ( see the top of `errand-envs.py` ).

| environment | where | toolchain |
|---|---|---|
| `mac_wheels` | here | clang from conda-forge |
| `lin_wheels` | a Linux machine with a GPU, in an Apptainer image ( `scripts/manylinux.def`, glibc 2.28, GCC 14 for nvcc ) | clang, nvcc |
| `win_wheels` | a Windows virtual machine booted by errand on a Linux host | llvm-mingw clang + GNU make ( `scripts/win_tools.ps1` ) |

`errand setup --env lin_wheels` and `errand setup --env win_wheels` build the images ( the Windows one takes about 45 minutes ).

The versions that matter are pinned in these environments. jax is `0.10.2` everywhere: the Jax kernels declare the XLA FFI
API version of the jaxlib they are compiled with, and a user's jaxlib takes them only if that version is not newer than its
own ( otherwise it compiles them, see `xla-ffi.json` in loom's `compilation/catalogue.py` ). An old jax does not slow the
kernels down.

To release, loom first ( each repository publishes its own package ):

```bash
cd ../loom && make version wheels && TWINE_PASSWORD=<your PyPI token> make publish     # when loom changed
cd ../sdot-ffi && python scripts/wheels.py --version next --no-build                    # sdot's version, and loom-ffi >= the checkout's
make wheels && TWINE_PASSWORD=<your PyPI token> make publish
```

`--version` writes sdot's version only, and makes it require at least the version of the loom checkout the kernels are built
with. A newer loom is fine: a kernel whose sources it generates otherwise is compiled.
