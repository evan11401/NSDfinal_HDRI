FLAGS ?=
CXX = g++
PY = python
PY_LIBS = $(shell python -m pybind11 --includes)
CPPFLAGS += `pkg-config --cflags --libs opencv` $(PY_LIBS) -shared -fPIC 

all: _align.so _calCRF.so _merge.so _tonemap.so

_align.so: mod/_align.cpp
	$(CXX) $< -o $@  $(CPPFLAGS) 

_calCRF.so: mod/_calCRF.cpp
	$(CXX) $< -o $@  $(CPPFLAGS) 

_merge.so: mod/_merge.cpp
	$(CXX) $< -o $@  $(CPPFLAGS) 

_tonemap.so: mod/_tonemap.cpp
	$(CXX) $< -o $@  $(CPPFLAGS) 


.PHONY: clean
clean:
	rm -rf  *.so
	rm -rf  *.ext
	rm -rf  *.jpg
