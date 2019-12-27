FLAGS ?=
CXX = g++
PY = python
#PY_LIBS = $(shell python3 -m pybind11 --includes)
PY_LIBS = -I/opt/conda/include/python3.6m 
CPPFLAGS += $(PY_LIBS) -O3 -shared -Wall -std=c++17 -g -m64 -fPIC `pkg-config --cflags --libs opencv4`

MKLROOT ?= 
MKLINC = -I$(MKLROOT)/include
LDFLAGS += -L$(MKLROOT)/lib -lmkl_rt -lpthread -lm -ldl

_align.so: mod/_align.cpp
	$(CXX) $(MKLINC) $(CPPFLAGS) $< -o $@ $(LDFLAGS)


.PHONY: clean test branch
clean:
	rm -rf  *.so
