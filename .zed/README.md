# Configuration Zed pour SDOT

## 🎯 Configuration automatique

Ce projet est configuré pour Zed avec :
- **clangd** pour C++ (parsing, autocomplétion, diagnostics)
- **pyright** pour Python (type checking, autocomplétion)
- **compile_commands.json** généré automatiquement par Meson
- **Tasks** prédéfinies pour build, test, etc.

## 📋 Tâches disponibles (Cmd+Shift+P → "tasks")

### Build
- 🔧 **Setup Meson** : Configure le projet (`meson setup build`)
- 🔨 **Build All** : Compile tout le projet
- 🔥 **Clean + Rebuild** : Nettoie et recompile

### Tests
- ✅ **Run All Tests** : Tous les tests (C++ + Python)
- 🧪 **C++ Tests Only** : Tests Catch2
- 🐍 **Python Tests Only** : Tests pytest
- 🔬 **PyTorch Tests** : Tests PyTorch uniquement
- 🎯 **JAX Tests** : Tests JAX uniquement

### Exemples
- 🏥 **CT Reconstruction Example** : Lance l'exemple de reconstruction

### Utilitaires
- 🧹 **Clean Build** : Nettoie les artefacts de build
- 📊 **Meson Configure** : Affiche la config Meson
- 🔍 **Check Compile Commands** : Vérifie compile_commands.json

## 🔧 LSP Configuration

### C++ (clangd)
- **Compile commands** : `./compile_commands.json` (symlink → `build/`)
- **Clang-tidy** : Activé (modernize, performance, readability, bugprone)
- **Inlay hints** : Noms de paramètres, types déduits
- **Index** : Background build pour recherche rapide

### Python (pyright)
- **Type checking** : Basic mode
- **Extra paths** : `python/pytorch`, `python/jax`, `build`
- **Venv** : `.venv` (détecté automatiquement)

## 📁 Fichiers de configuration

```
.zed/
├── settings.json      # Configuration LSP + éditeur
├── tasks.json         # Tâches de build/test
└── README.md          # Ce fichier

.clangd                # Configuration clangd (clang-tidy, hints)
pyrightconfig.json     # Configuration pyright (type checking)
compile_commands.json  # Symlink vers build/ (généré par Meson)
```

## 🚀 Quick Start

1. **Première compilation**
   ```bash
   make setup && make build
   ```

2. **Ouvrir dans Zed**
   ```bash
   zed .
   ```

3. **Vérifier que les LSP fonctionnent**
   - Ouvrir `src/cpu/sdot_w2_cpu.cpp` → clangd devrait activer
   - Ouvrir `python/pytorch/sdot_pytorch.py` → pyright devrait activer
   - Vérifier la barre de statut en bas (icônes LSP)

4. **Lancer les tests** (Cmd+Shift+P → "tasks" → "Run All Tests")

## 🐛 Troubleshooting

### clangd ne trouve pas les headers
```bash
# Régénérer compile_commands.json
rm -rf build
make setup && make build
```

### pyright ne trouve pas les modules
```bash
# Vérifier que le venv est activé
ls .venv/bin/python

# Reconstruire les modules
make build
```

### Tâches Zed ne fonctionnent pas
```bash
# Vérifier que Meson est installé dans le venv
.venv/bin/meson --version

# Réinstaller si nécessaire
.venv/bin/pip install meson
```

## 💡 Tips

- **Rebuild après changement C++** : `Cmd+Shift+P` → "Build All"
- **Jump to definition** : `Cmd+Click` ou `F12`
- **Find all references** : `Shift+F12`
- **Rename symbol** : `F2`
- **Format code** : Désactivé (utilisez clang-format manuellement si besoin)

## 🔗 Liens utiles

- [Zed Documentation](https://zed.dev/docs)
- [clangd Documentation](https://clangd.llvm.org/)
- [Pyright Documentation](https://github.com/microsoft/pyright)
