FLAGS ?=
CXX = g++
PY = python
#PY_LIBS = $(shell python -m pybind11 --includes)
PY_LIBS = -I/home/evan11401/Desktop/Courses/nsd/final/NSDfinal_HDRI/projectenv/include/python3.6m
#CPPFLAGS += $(PY_LIBS) -O3 -shared -Wall -std=c++17 -g -m64 -fPIC `pkg-config --cflags --libs opencv`
CPPFLAGS += `pkg-config --cflags --libs opencv` $(PY_LIBS) -shared -fPIC 


MKLROOT ?= 
MKLINC = -I$(MKLROOT)/include
LDFLAGS += -L$(MKLROOT)/lib -lmkl_rt -lpthread -lm -ldl

all: _align.so _calCRF.so _merge.so

_align.so: mod/_align.cpp
	$(CXX) $< -o $@ $(MKLINC) $(CPPFLAGS)  $(LDFLAGS)

_calCRF.so: mod/_calCRF.cpp
	$(CXX) $< -o $@ $(MKLINC) $(CPPFLAGS)  $(LDFLAGS)

_merge.so: mod/_merge.cpp
	$(CXX) $< -o $@ $(MKLINC) $(CPPFLAGS)  $(LDFLAGS)



.PHONY: clean test branch
clean:
	rm -rf  *.so
	rm -rf  *.ext
