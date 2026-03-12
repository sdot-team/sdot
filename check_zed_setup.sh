#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         🔍 Vérification Configuration Zed pour SDOT              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check compile_commands.json
echo "📝 Vérification compile_commands.json..."
if [ -L compile_commands.json ]; then
    check_pass "compile_commands.json (symlink → build/)"
else
    check_fail "compile_commands.json manquant (run: make build)"
fi

# Check Zed config files
echo
echo "⚙️  Vérification fichiers Zed..."
if [ -f .zed/settings.json ]; then
    check_pass ".zed/settings.json"
else
    check_fail ".zed/settings.json manquant"
fi

if [ -f .zed/tasks.json ]; then
    check_pass ".zed/tasks.json"
else
    check_fail ".zed/tasks.json manquant"
fi

# Check clangd config
echo
echo "🔧 Vérification clangd..."
if [ -f .clangd ]; then
    check_pass ".clangd configuration"
else
    check_warn ".clangd manquant (optionnel mais recommandé)"
fi

# Check pyright config
echo
echo "🐍 Vérification pyright..."
if [ -f pyrightconfig.json ]; then
    check_pass "pyrightconfig.json"
else
    check_warn "pyrightconfig.json manquant (optionnel)"
fi

# Check Python venv
echo
echo "🐍 Vérification Python venv..."
if [ -d .venv ]; then
    check_pass ".venv existe"
    if [ -f .venv/bin/python ]; then
        PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
        check_pass "Python: $PYTHON_VERSION"
    else
        check_fail ".venv/bin/python manquant"
    fi
else
    check_fail ".venv manquant (run: make venv)"
fi

# Check Meson
echo
echo "🔨 Vérification Meson..."
if [ -f .venv/bin/meson ]; then
    MESON_VERSION=$(.venv/bin/meson --version 2>&1)
    check_pass "Meson: $MESON_VERSION"
else
    check_fail "Meson manquant (run: .venv/bin/pip install meson)"
fi

# Check build directory
echo
echo "🏗️  Vérification build..."
if [ -d build ]; then
    check_pass "build/ existe"
    if [ -f build/compile_commands.json ]; then
        COMPILE_CMDS=$(wc -l < build/compile_commands.json)
        check_pass "compile_commands.json: $COMPILE_CMDS lignes"
    else
        check_fail "build/compile_commands.json manquant (run: make build)"
    fi
else
    check_fail "build/ manquant (run: make setup && make build)"
fi

# Check Python modules
echo
echo "📦 Vérification modules Python..."
PYTORCH_MODULE=$(ls python/pytorch/*.so 2>/dev/null | wc -l)
JAX_MODULE=$(ls python/jax/*.so 2>/dev/null | wc -l)

if [ "$PYTORCH_MODULE" -gt 0 ]; then
    check_pass "sdot_pytorch_cpp.so compilé"
else
    check_fail "sdot_pytorch_cpp.so manquant (run: make build)"
fi

if [ "$JAX_MODULE" -gt 0 ]; then
    check_pass "sdot_jax_cpp.so compilé"
else
    check_fail "sdot_jax_cpp.so manquant (run: make build)"
fi

# Check clangd availability
echo
echo "🔍 Vérification outils LSP..."
if command -v clangd &> /dev/null; then
    CLANGD_VERSION=$(clangd --version 2>&1 | head -1)
    check_pass "clangd disponible: $CLANGD_VERSION"
else
    check_warn "clangd non trouvé dans PATH (installé via Homebrew?)"
fi

echo
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ Configuration OK!                             ║"
echo "║                                                                  ║"
echo "║  Vous pouvez maintenant ouvrir le projet dans Zed:              ║"
echo "║                                                                  ║"
echo "║    $ zed .                                                       ║"
echo "║                                                                  ║"
echo "║  Puis appuyez sur Cmd+Shift+P → \"tasks\" pour voir les tâches    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo
