# SO := generated/bsp_FP64_2_cpu.$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX')[1:])")

XMAKE_ENV := \
	SDOT_NB_INCLUDE=$(SDOT_NB_INCLUDE) \
	SDOT_NB_LIB=$(SDOT_NB_LIB) \
	SDOT_BINDING_NAME=bsp_FP64_2_cpu \
	SDOT_SRC_INCLUDE=../../.. \
	SDOT_OUTPUT_DIR=generated \
	SDOT_SRC_FILES=generated/sources/bsp_FP64_2_cpu.cpp,../../../cpp/geometry/SimpleSquareMatrix_eigen.cpp,../../../cpp/geometry/VtkOutput.cpp \
	SDOT_ARCH=cpu \
	SDOT_DEFINES=SDOT_NANOBIND_ARCH=cpu,SDOT_SCALAR_TYPE=FP64

all:
	$(XMAKE_ENV) micromamba -n vfs run xmake f -yc
	$(XMAKE_ENV) micromamba -n vfs run xmake
	$(XMAKE_ENV) PYTHONPATH=../.. micromamba -n vfs run python3 test.py 2>&1


.PHONY: all
