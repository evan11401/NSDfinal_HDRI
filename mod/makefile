FLAGS ?=
CXX = g++
PY = python
PY_LIBS = $(shell python3 -m pybind11 --includes)
CPPFLAGS += $(PY_LIBS) -O3 -shared -Wall -std=c++17 -g -m64 -fPIC

MKLROOT ?= ${HOME}/opt/conda
MKLINC = -I$(MKLROOT)/include
LDFLAGS += -L$(MKLROOT)/lib -lmkl_rt -lpthread -lm -ldl

_align.so: _align.cpp
	$(CXX) $(MKLINC) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean test branch
clean:
	rm -rf  *.so
test: _matrix.so
	$(PY) test.py
bench: _matrix.so
	$(PY) bench.py