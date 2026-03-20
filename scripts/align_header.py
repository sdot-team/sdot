#!/usr/bin/env python3
"""
align_header.py — aligne les noms de déclarations C++ dans un fichier .h

Utilise `clang -ast-dump=json` pour trouver la colonne exacte de chaque nom
de déclaration (using, attribut, méthode, constructeur), puis aligne tous les
noms d'un même scope à la même colonne.

Usage (standalone) :
    python3 scripts/align_header.py [--inplace] file.h [-- <clang-flags>]

Usage (VS Code via emeraldwalk.runonsave) :
    configuré dans .vscode/settings.json
"""

from __future__ import annotations
import sys, os, json, subprocess
from pathlib import Path
from collections import defaultdict

# ── Exécutable clang ──────────────────────────────────────────────────────────

_CLANG_CANDIDATES = [
    '/opt/homebrew/Cellar/llvm/21.1.8_1/bin/clang',
    '/opt/homebrew/bin/clang',
    '/usr/local/bin/clang',
    '/usr/bin/clang',
]
CLANG = next((p for p in _CLANG_CANDIDATES if os.path.exists(p)), 'clang')

# ── Types AST ─────────────────────────────────────────────────────────────────

# Nœuds dont on veut aligner le nom
DECL_KINDS = frozenset({
    'TypeAliasDecl',       # using X = ...
    'FieldDecl',           # attribut de classe / struct
    'VarDecl',             # variable
    'FunctionDecl',        # fonction libre
    'CXXMethodDecl',       # méthode membre
    'CXXConstructorDecl',  # constructeur
    'CXXDestructorDecl',   # destructeur
})

# Nœuds qui ouvrent un nouveau scope d'alignement
SCOPE_KINDS = frozenset({
    'TranslationUnitDecl',
    'NamespaceDecl',
    'ClassTemplateDecl',
    'CXXRecordDecl',       # class, struct, union
})

# ── Étape 1 : obtenir l'AST ───────────────────────────────────────────────────

def get_ast(path: Path, extra_flags: list[str]) -> dict | None:
    # Heuristique : inclure le répertoire du fichier et ses parents jusqu'à src/
    auto_inc = []
    d = path.parent
    for _ in range(5):
        auto_inc += ['-I', str(d)]
        if d.name in ('src', 'include'):
            break
        d = d.parent

    cmd = [
        CLANG,
        '-fsyntax-only',
        '-Xclang', '-ast-dump=json',  # -ast-dump=json via -Xclang pour compatibilité macOS
        '-std=c++20', '-x', 'c++',
        '-w',            # pas de warnings
        *auto_inc,
        *extra_flags,
        str(path),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None

# ── Étape 2 : extraire les déclarations par scope ────────────────────────────

def collect(node: dict, target: str,
            scope: str = '__root__',
            by_scope: dict | None = None,
            cur_file: str | None = None) -> dict:
    """
    Parcourt l'AST et remplit by_scope :
        scope_id → [{line, col, name, kind}, ...]

    `col` est la colonne (1-indexée) du nom de la déclaration,
    telle que rapportée par clang.
    """
    if by_scope is None:
        by_scope = defaultdict(list)

    kind     = node.get('kind', '')
    loc      = node.get('loc',  {})
    cur_file = loc.get('file', cur_file)   # clang n'émet 'file' que lors d'un changement

    # Enregistrer si la déclaration est dans notre fichier cible
    if kind in DECL_KINDS and cur_file == target:
        line = loc.get('line', 0)
        col  = loc.get('col',  0)
        name = node.get('name', '')
        # Les constructeurs/destructeurs de template ont un nom de la forme
        # "Cell<TF, _dim>" dans l'AST — on garde uniquement l'identifiant de base
        if '<' in name:
            name = name[:name.index('<')]
        if line and col and name:
            by_scope[scope].append(dict(line=line, col=col, name=name, kind=kind))

    # Descente récursive en ouvrant un nouveau scope si nécessaire
    child_scope = node.get('id', kind) if kind in SCOPE_KINDS else scope
    for child in node.get('inner', []):
        collect(child, target, child_scope, by_scope, cur_file)

    return by_scope

# ── Étape 3 : ré-aligner les lignes ──────────────────────────────────────────

def apply_alignment(source: str, by_scope: dict) -> str:
    lines = source.splitlines(keepends=True)

    for scope, decls in by_scope.items():
        if len(decls) < 2:
            continue

        max_col = max(d['col'] for d in decls)

        for d in decls:
            gap = max_col - d['col']
            if gap == 0:
                continue

            li   = d['line'] - 1   # index ligne (0-based)
            pos  = d['col']  - 1   # index char  (0-based)
            name = d['name']
            line = lines[li]

            # Vérification : le nom doit être exactement à la position attendue
            if line[pos:pos + len(name)] != name:
                continue

            lines[li] = line[:pos] + ' ' * gap + line[pos:]
            d['col'] += gap   # cohérence si la même ligne est revisitée

    return ''.join(lines)

# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    argv = sys.argv[1:]

    inplace = '--inplace' in argv
    if inplace:
        argv.remove('--inplace')

    if '--' in argv:
        i           = argv.index('--')
        files       = argv[:i]
        extra_flags = argv[i + 1:]
    else:
        files, extra_flags = argv, []

    if not files:
        sys.exit(f'usage: {sys.argv[0]} [--inplace] <file.h> [-- <clang-flags>]')

    for f in files:
        path   = Path(f).resolve()
        source = path.read_text()

        ast = get_ast(path, extra_flags)
        if ast is None:
            sys.stderr.write(f'warning: impossible de parser {f}, fichier inchangé\n')
            if not inplace:
                sys.stdout.write(source)
            continue

        by_scope = collect(ast, str(path))
        result   = apply_alignment(source, by_scope)

        if inplace:
            path.write_text(result)
        else:
            sys.stdout.write(result)


if __name__ == '__main__':
    main()
