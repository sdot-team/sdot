# Releasing sdot

sdot is released with loom's commands ( `python -m loom.release`, see loom's guide, *Releasing a package* ), from the root of
the checkout, the loom checkout beside it; the `Makefile` has one-line reminders of them. What is particular to sdot is in
`[tool.loom.release]` of `pyproject.toml`.

To release, loom first ( each repository publishes its own package, and sdot requires at least the loom its kernels are built
with ):

```bash
cd ../loom && make bump_version && make wheels && TWINE_PASSWORD=<your PyPI token> make publish     # when loom changed
cd ../sdot-ffi && make bump_version          # sdot's version, and loom-ffi >= the checkout's
make wheels_all && TWINE_PASSWORD=<your PyPI token> make publish
```

`make publish` refuses while the loom sdot requires is not on PyPI.

`make wheels` builds the sdot wheel of this machine. `make wheels_all` ( `python -m loom.release dispatch_wheels all` ) builds
every sdot wheel and brings them into `dist/`: it runs `python -m loom.release make_wheels` in loom's build environment of each
platform of `[tool.loom.release] platforms` ( `loom-macos-arm64`, `loom-linux-x86_64`, `loom-windows-x86_64`, see loom's
`loom/release/envs.py` ), through errand, on its own copy of the tree: the kernels of every framework it has
( `python -m sdot.catalogue` per framework ), then its sdot wheel, validated in a clean virtual environment ( with a wheel of
the loom checkout, built on the side and not kept ): `scripts/smoke.py`, then the documented snippets with
`LOOM_KERNELS=catalogue`. Some platforms only: `make wheels_all PLATFORMS=linux-x86_64,windows-x86_64`.

The x86 levels a wheel carries are decided by measure while its kernels are built ( `level_workloads` in
`src/sdot/catalogue.py`, `levels.json` in the catalogue ): run it on a quiet machine, or a level that gains a little is dropped
as noise.

Where the machines are is in `errand-vars.py`, which is not versioned ( `loom_linux_host`, `loom_linux_root`,
`loom_linux_scratch`, `loom_linux_cache`, `loom_linux_binds`, `loom_windows_host`, `loom_windows_root`, `loom_mamba_root` ):

| platform | where | toolchain ( loom's recipes ) |
|---|---|---|
| `macos-arm64` | here | clang from conda-forge |
| `linux-x86_64` | `loom_linux_host`, in an Apptainer image ( manylinux_2_28, GCC 14 for nvcc ) | clang, nvcc |
| `windows-x86_64` | a Windows virtual machine booted by errand on `loom_windows_host` | llvm-mingw clang + GNU make |

`python -m loom.release errand setup --env loom-linux-x86_64,loom-windows-x86_64` builds the images ( the Windows one takes
about 45 minutes ): errand, with loom's environments declared.

The versions that matter are pinned in these environments. jax is `0.10.2` everywhere: the Jax kernels declare the XLA FFI
API version of the jaxlib they are compiled with, and a user's jaxlib takes them only if that version is not newer than its
own ( otherwise it compiles them, see `xla-ffi.json` in loom's `compilation/catalogue.py` ). An old jax does not slow the
kernels down. A newer loom than the one sdot requires is fine: a kernel whose sources it generates otherwise is compiled.
