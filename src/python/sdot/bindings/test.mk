
SO := generated/bsp_FP64_2_cpu.$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX')[1:])")

# Build (or reuse) libnanobind_sdot.a from the active Python's nanobind sources
NB_CACHE_ID  := $(shell python3 -c "import hashlib,sys,importlib.util,sysconfig,platform; spec=importlib.util.find_spec('nanobind'); nb=spec.submodule_search_locations[0] if spec.submodule_search_locations else __import__('pathlib').Path(spec.origin).parent; key=f'{nb}|{sys.executable}|{platform.mac_ver()[0]}'; print(hashlib.sha1(key.encode()).hexdigest()[:12])")
NB_SRC       := $(shell python3 -c "import importlib.util,pathlib; spec=importlib.util.find_spec('nanobind'); print(pathlib.Path(spec.origin).parent)")
PYINC        := $(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYVER        := $(shell python3 -c "import sysconfig; print(sysconfig.get_python_version())")
NB_CACHE_DIR := $(HOME)/.cache/sdot/nanobind-$(PYVER)-$(NB_CACHE_ID)
NB_LIB_FILE  := $(NB_CACHE_DIR)/libnanobind_sdot.a

SDOT_NB_INCLUDE := $(PYINC),$(NB_SRC)/include,$(NB_SRC)/ext/robin_map/include
SDOT_NB_LIB     := $(NB_CACHE_DIR):nanobind_sdot

XMAKE_ENV := \
	SDOT_NB_INCLUDE=$(SDOT_NB_INCLUDE) \
	SDOT_NB_LIB=$(SDOT_NB_LIB) \
	SDOT_BINDING_NAME=bsp_FP64_2_cpu \
	SDOT_SRC_INCLUDE=../../.. \
	SDOT_OUTPUT_DIR=generated \
	SDOT_SRC_FILES=generated/sources/bsp_FP64_2_cpu.cpp,../../../cpp/geometry/SimpleSquareMatrix_eigen.cpp,../../../cpp/geometry/VtkOutput.cpp \
	SDOT_ARCH=cpu \
	SDOT_DEFINES=SDOT_NANOBIND_ARCH=cpu,SDOT_SCALAR_TYPE=FP64

$(NB_LIB_FILE):
	@echo "=== building libnanobind_sdot.a for Python $(PYVER) ==="
	mkdir -p $(NB_CACHE_DIR)
	clang++ -std=c++20 -O2 \
	  -I$(PYINC) -I$(NB_SRC)/include -I$(NB_SRC)/ext/robin_map/include \
	  -fPIC -fvisibility=hidden -mmacosx-version-min=14.0 \
	  -c $(NB_SRC)/src/nb_combined.cpp -o $(NB_CACHE_DIR)/nb_combined.o
	ar rcs $(NB_LIB_FILE) $(NB_CACHE_DIR)/nb_combined.o

all: build diag test

build: $(NB_LIB_FILE)
	$(XMAKE_ENV) xmake f -y
	$(XMAKE_ENV) xmake

diag: $(SO)
	@echo "=== library deps (otool -L) ==="
	otool -L $(SO)
	@echo "=== rpaths (LC_RPATH) ==="
	otool -l $(SO) | grep -A2 LC_RPATH || echo "(aucun rpath)"
	@echo "=== symboles PyInit ==="
	nm -g $(SO) | grep PyInit
	@echo "=== python utilisé ==="
	python3 --version
	python3 -c "import sys; print(sys.executable)"

test:
	@echo "=== dlopen trace ==="
	DYLD_PRINT_LIBRARIES=1 PYTHONPATH=../.. python3 -c \
	  "import sdot.bindings.generated.bsp_FP64_2_cpu as m; print('OK:', m)" 2>&1 \
	  | grep -iE "bsp_FP64|libz|error|OK:" || true
	@echo "=== exception avant SystemError ==="
	PYTHONPATH=../.. python3 test.py 2>&1
	@echo "=== sans directive #line ==="
	sed '1{/^#line/d}' generated/sources/bsp_FP64_2_cpu.cpp > /tmp/bsp_noline.cpp
	diff -u generated/sources/bsp_FP64_2_cpu.cpp /tmp/bsp_noline.cpp | head -5
	$(XMAKE_ENV) SDOT_SRC_FILES=/tmp/bsp_noline.cpp,../../../cpp/geometry/SimpleSquareMatrix_eigen.cpp,../../../cpp/geometry/VtkOutput.cpp xmake 2>&1 | tail -3
	PYTHONPATH=../.. python3 -c \
	  "import sdot.bindings.generated.bsp_FP64_2_cpu as m; print('OK sans #line:', m)" 2>&1

minimal: $(NB_LIB_FILE)
	@echo "=== compilation d'un module nanobind minimal ==="
	printf '#include <nanobind/nanobind.h>\nNB_MODULE(nb_minimal, m) { m.attr("ok") = 42; }\n' > /tmp/nb_minimal.cpp
	$(XMAKE_ENV) SDOT_BINDING_NAME=nb_minimal \
	             SDOT_SRC_FILES=/tmp/nb_minimal.cpp \
	             xmake f -y 2>&1 | tail -3
	$(XMAKE_ENV) SDOT_BINDING_NAME=nb_minimal \
	             SDOT_SRC_FILES=/tmp/nb_minimal.cpp \
	             xmake 2>&1 | tail -5
	PYTHONPATH=../.. python3 -c "import sdot.bindings.generated.nb_minimal as m; print('minimal OK:', m.ok)"

fix:
	@echo "=== suppression du cache nanobind_sdot ==="
	rm -f $(NB_LIB_FILE)
	@echo "=== reconstruction + test ==="
	$(MAKE) -f test.mk build test

.PHONY: all build diag test minimal fix
