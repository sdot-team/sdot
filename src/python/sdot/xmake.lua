-- ============================================================
--  xmake.lua — binding Python sdot
-- ============================================================

add_rules( "mode.release", "mode.debug" )
set_defaultmode( "release" )
-- set_defaultmode( "debug" )

-- ──────────────────────────── dépendances ────────────────────────────────────

add_requires( "zpp_bits" )
add_requires( "eigen" )

-- ──────────────────────────── helper ─────────────────────────────────────────

local function split_env( name )
    local val = os.getenv( name ) or ""
    local t = {}
    for v in val:gmatch("[^,]+") do t[ #t + 1 ] = v end
    return t
end

-- ──────────────────────────── target ─────────────────────────────────────────

target( os.getenv( "SDOT_BINDING_NAME" ) )
    set_kind( "shared" )
    add_rules( "python.module" )
    set_languages( "cxx20" )
    set_targetdir( os.getenv( "SDOT_OUTPUT_DIR" ) )

    arch = os.getenv( "SDOT_ARCH" )
    if arch == "cuda" then
        add_rules( "cuda.build" )
        add_cuflags( "-diag-suppress 1160", { force = true } )
        add_cuflags( "--expt-relaxed-constexpr" )
        add_cuflags( "--use_fast_math" )
        add_cuflags( "--extended-lambda" )
    end

    for _, f in ipairs( split_env( "SDOT_SRC_FILES" ) ) do
        add_files( f )
    end

    -- Add nanobind source
    add_files( os.getenv( "SDOT_NANOBIND_SRC" ), { cucompiler = false } )

    add_defines( "SDOT_BINDING_NAME=" .. os.getenv( "SDOT_BINDING_NAME" ) )
    for _, f in ipairs( split_env("SDOT_DEFINES" ) ) do
        add_defines( f )
    end

    add_cxxflags( "-fdiagnostics-absolute-paths" )
    for _, f in ipairs( split_env("SDOT_EXTRA_CFLAGS" ) ) do
        add_cxxflags(f)
    end

    -- nanobind recommends -fno-strict-aliasing
    add_cxxflags( "-fno-strict-aliasing" )
    add_cxxflags( "-g3" )
    if is_mode("release") then
        add_defines("NB_COMPACT_ASSERTIONS")
    end

    -- Add packages
    add_packages( "zpp_bits", "eigen" )

    on_load( function(target)
        target:add("includedirs", os.getenv( "SDOT_SRC_INCLUDE" ))
        target:add("includedirs", os.getenv( "SDOT_NANOBIND_INC" ))
        target:add("includedirs", os.getenv( "SDOT_ROBIN_MAP_INC" ))
        target:add("sysincludedirs", os.getenv( "SDOT_PYTHON_INC" ))

        if target:is_plat("macosx") then
            --  target:add("cxflags", "-mmacosx-version-min=11.0")
            --  target:add("ldflags", "-mmacosx-version-min=11.0")
            --  target:add("shflags", "-mmacosx-version-min=11.0")
             target:add("ldflags", "-Wl,-undefined,dynamic_lookup")
             target:add("shflags", "-Wl,-undefined,dynamic_lookup")
        end
    end )

    -- before_build( function( target )
    --     import( "core.project.task" )
    --     local project_root = path.directory( os.getenv( "SDOT_SRC_INCLUDE" ) )
    --     task.run( "project", { kind = "compile_commands", outputdir = project_root } )
    -- end )
target_end()
