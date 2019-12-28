
#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


#include <pybind11/stl.h>

#include <mkl.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <functional>


namespace py = pybind11;
using namespace cv;
using namespace std;
