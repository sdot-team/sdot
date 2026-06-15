def axis_tag_decls( names ) -> list[ str ]:
    """C++ declarations of the empty marker structs `struct ax_<name> {};` that name batch axes.

    Each axis name yields a tag struct used both as `container_tags::AxisNames< ax_… >` on tensor
    types and as the `Ax...` a `Parallel_*` functor is instantiated with. The same name can surface
    in several generated headers and in the binding, so every declaration is wrapped in an include
    guard — declaring a tag twice is harmless, the guard just makes it a no-op.
    """
    lines: list[ str ] = []
    for name in names:
        guard = f"SDOT_AXIS_NAME_ax_{ name }"
        lines += [ f"#ifndef { guard }", f"#define { guard }", f"struct ax_{ name } {{}};", "#endif" ]
    return lines
