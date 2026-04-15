-- ==================================================================
--  reusable xmake.lua — most of the content comes from env variables
-- ==================================================================

add_rules("mode.release", "mode.debug")
set_defaultmode("release")

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

target(os.getenv("SDOT_XMAKE_TARGET") or "sdot_binding")
    set_targetdir(os.getenv("SDOT_XMAKE_OUTPUT_DIR") or "build")
    add_rules("python.module")
    set_languages("cxx20")
    set_kind("shared")

    -- debug / release changes
    if is_mode("release") then
        add_defines("NB_COMPACT_ASSERTIONS")
    else
        set_symbols("debug")
    end

    -- CUDA specificities
    if os.getenv("SDOT_XMAKE_NEEDS_CUDA") == "1" then
        add_rules("cuda.build")
        add_cuflags("-diag-suppress 1160", {force = true})
        add_cuflags("--expt-relaxed-constexpr", "--use_fast_math", "--extended-lambda")
    end

    -- MacOS specificities
    if is_plat("macosx") then
        add_ldflags("-Wl,-undefined,dynamic_lookup")
        add_shflags("-Wl,-undefined,dynamic_lookup")
    end

    -- generic variables
    add_includedirs(get_env_list("SDOT_XMAKE_INCLUDES"))
    add_cxxflags(get_env_list("SDOT_XMAKE_CXXFLAGS"))
    add_defines(get_env_list("SDOT_XMAKE_DEFINES"))
    add_files(get_env_list("SDOT_XMAKE_SOURCES"))
    add_packages(requires)
