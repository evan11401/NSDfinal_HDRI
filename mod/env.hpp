
#include <opencv2/opencv.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>


#include <pybind11/stl.h>

#include <mkl.h>

#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <functional>


namespace py = pybind11;
using namespace cv;

Mat triangleWeights()
{
    // hat function
    Mat w(LDR_SIZE, 1, CV_32F);
    int half = LDR_SIZE / 2;
    for(int i = 0; i < LDR_SIZE; i++) {
        w.at<float>(i) = i < half ? i + 1.0f : LDR_SIZE - i;
    }
    return w;
}
void checkImageDimensions(const std::vector<Mat>& images)
{
    CV_Assert(!images.empty());
    int width = images[0].cols;
    int height = images[0].rows;
    int type = images[0].type();

    for(size_t i = 0; i < images.size(); i++) {
        CV_Assert(images[i].cols == width && images[i].rows == height);
        CV_Assert(images[i].type() == type);
    }
}
Mat linearResponse(int channels)
{
    Mat response = Mat(LDR_SIZE, 1, CV_MAKETYPE(CV_32F, channels));
    for(int i = 0; i < LDR_SIZE; i++) {
        response.at<Vec3f>(i) = Vec3f::all(static_cast<float>(i));
    }
    return response;
}

std::vector<Mat> images_pytocpp(std::vector<py::array_t<unsigned char>> src)
{
    std::vector<Mat> mat_src;
    mat_src.resize(0);
    for(uint i=0;i<src.size();i++){
        py::buffer_info buf = src[i].request();
        Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
        mat_src.push_back(mat);
    }
    return mat_src;
}
std::vector<float> times_pytocpp(py::array_t<float> times)
{
    std::vector<float> time_vec;
    time_vec.resize(0);
    py::buffer_info time_buf = times.request();
    float *time_ptr = (float *) time_buf.ptr;
    for(uint i=0;i<time_buf.shape[0];i++){        
        time_vec.push_back(time_ptr[i]);
    }
    return time_vec;
}
Mat debs_pytocpp(py::array_t<float> resDebevec){
    py::buffer_info buf = resDebevec.request();
    Mat mat(buf.shape[0], buf.shape[1], CV_32FC3, (float*)buf.ptr);
    return mat;
}