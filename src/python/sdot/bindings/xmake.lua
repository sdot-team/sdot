-- ============================================================
--  xmake.lua — binding Python sdot
--  Toute la configuration est lue depuis des variables d'env :
--    SDOT_BINDING_NAME  : nom du module Python à générer
--    SDOT_OUTPUT_DIR    : répertoire de sortie du .so
--    SDOT_SRC_FILES     : sources .cpp séparées par ","
--    SDOT_SRC_INCLUDE   : chemin vers src/ du projet
--    SDOT_EXTRA_CFLAGS  : flags C++ supplémentaires séparés par "," (optionnel)
-- ============================================================

add_rules( "mode.release", "mode.debug" )
set_defaultmode( "release" )

-- ──────────────────────────── dépendances ────────────────────────────────────

-- If SDOT_NB_INCLUDE / SDOT_NB_LIB are set we use a locally-built nanobind
-- (compiled against the active Python) instead of the xmake-managed package.
local _nb_include = os.getenv( "SDOT_NB_INCLUDE" )
local _nb_lib     = os.getenv( "SDOT_NB_LIB" )

if not _nb_include then
    add_requires( "nanobind" )
end
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
        -- set_toolchains("cuda")
        -- add_extsources( "cuda", ".cpp" )
        -- add_cuflags( "-arch=native" )
        add_cuflags( "-diag-suppress 1160", { force = true } )
        add_cuflags( "--expt-relaxed-constexpr" )
        add_cuflags( "--use_fast_math" )
        add_cuflags( "--extended-lambda" )
        -- add_packages("cuda")
    end

    for _, f in ipairs( split_env( "SDOT_SRC_FILES" ) ) do
        add_files( f )
    end

    add_defines( "SDOT_BINDING_NAME=" .. os.getenv( "SDOT_BINDING_NAME" ) )
    for _, f in ipairs( split_env("SDOT_DEFINES" ) ) do
        add_defines( f )
    end

    -- add_cxxflags("-fvisibility=hidden", "-fvisibility-inlines-hidden")
    add_cxxflags( "-fdiagnostics-absolute-paths" )
    for _, f in ipairs( split_env("SDOT_EXTRA_CFLAGS" ) ) do
        add_cxxflags(f)
    end

    if _nb_include then
        add_packages( "zpp_bits", "eigen" )
    else
        add_packages( "nanobind", "zpp_bits", "eigen" )
    end

    on_load( function(target)
        target:add("includedirs", os.getenv( "SDOT_SRC_INCLUDE" ))
        if _nb_include then
            target:add("includedirs", _nb_include:split(","))
            -- Python symbols are resolved at runtime via the interpreter
            target:add("shflags", "-undefined dynamic_lookup", { force = true })
            if _nb_lib then
                -- _nb_lib is "dir:libname", e.g. "/tmp/nb_build:nanobind_mamba"
                local parts = _nb_lib:split(":")
                target:add("linkdirs", parts[1])
                target:add("links",    parts[2])
            end
        end
    end )

    before_build( function( target )
        import( "core.project.task" )
        -- SDOT_SRC_INCLUDE pointe vers project_root/src, on remonte d'un cran
        local project_root = path.directory( os.getenv( "SDOT_SRC_INCLUDE" ) )
        task.run( "project", { kind = "compile_commands", outputdir = project_root } )
    end )
target_end()

