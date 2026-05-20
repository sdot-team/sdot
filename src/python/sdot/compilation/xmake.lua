-- ==================================================================
--  reusable xmake.lua — most of the content comes from env variables
-- ==================================================================

add_rules("mode.release", "mode.debug")
set_defaultmode("release")
-- set_defaultmode("debug")

-- ──────────────────────────── helpers ─────────────────────────────────────────

local function get_env_list(name)
    local val = os.getenv(name) or ""
    local t = {}
    for v in val:gmatch("[^,]+") do
        table.insert(t, v:trim())
    end
    return t
end

-- ──────────────────────────── dépendances ────────────────────────────────────

local requires = get_env_list("SDOT_XMAKE_REQUIRES")
add_requires(requires)

-- ──────────────────────────── target ─────────────────────────────────────────

-- "shared" -> python extension module (bindings) ; "binary" -> standalone executable (tests)
local target_kind = os.getenv("SDOT_XMAKE_KIND") or "shared"

target(os.getenv("SDOT_XMAKE_TARGET") or "sdot_binding")
    set_targetdir(os.getenv("SDOT_XMAKE_OUTPUT_DIR") or "build")
    set_languages("cxx20")
    if target_kind == "binary" then
        set_kind("binary")
    else
        add_rules("python.module")
        set_kind("shared")
    end

    -- debug / release changes
    if is_mode("release") then
        add_defines("NB_COMPACT_ASSERTIONS")
        -- add_cxxflags("-Wnan-infinity-disabled")
        add_cxxflags("-march=native")
        add_cxxflags("-ffast-math")
        add_cxxflags("-O3")
    else
        set_symbols("debug")
        add_cxxflags("-g3")
    end

    -- CUDA specificities
    if os.getenv("SDOT_XMAKE_NEEDS_CUDA") == "1" then
        add_rules("cuda.build")
        add_cuflags("-diag-suppress 940", {force = true})
        add_cuflags("-diag-suppress 1160", {force = true})
        add_cuflags("-diag-suppress 2473", {force = true})
        add_cuflags("--expt-relaxed-constexpr", "--use_fast_math", "--extended-lambda", "-extended-lambda")
        -- add_cuflags("-Wnan-infinity-disabled")
    else
        add_cxxflags("-fdiagnostics-absolute-paths")
    end

    -- MacOS specificities (undefined symbols resolved at load time — only for the python module)
    if is_plat("macosx") and target_kind ~= "binary" then
        add_ldflags("-Wl,-undefined,dynamic_lookup")
        add_shflags("-Wl,-undefined,dynamic_lookup")
    end

    -- generic variables
    add_includedirs(get_env_list("SDOT_XMAKE_INCLUDES"))
    add_cxxflags(get_env_list("SDOT_XMAKE_CXXFLAGS"))
    add_defines(get_env_list("SDOT_XMAKE_DEFINES"))
    add_files(get_env_list("SDOT_XMAKE_SOURCES"))
    add_packages(requires)
