#include "env.hpp"

Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input);

Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input);

py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(Mat & input);

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(Mat & input);