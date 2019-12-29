#include "env.hpp"


class Merge
{
public:
    Merge() :
        name("MergeDebevec"),
        weights(triangleWeights())
    {
    }

    void process(std::vector<Mat> images, std::vector<float> times, Mat response)
    {
        
        int channels = images[0].channels();
        Size size = images[0].size();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);        

        if(response.empty()) {
            response = linearResponse(channels);
            response.at<Vec3f>(0) = response.at<Vec3f>(1);
        }

        Mat log_response;
        log(response, log_response);        
        for(int i=0;i<times.size();i++)
            times[i] = std::log(times[i]);

        Mat result = Mat::zeros(size, CV_32FCC);
        std::vector<Mat> result_split;
        split(result, result_split);
        Mat weight_sum = Mat::zeros(size, CV_32F);

        for(size_t i = 0; i < images.size(); i++) {
            std::vector<Mat> splitted;
            split(images[i], splitted);

            Mat w = Mat::zeros(size, CV_32F);
            for(int c = 0; c < channels; c++) {
                LUT(splitted[c], weights, splitted[c]);
                w += splitted[c];
            }
            w /= channels;

            Mat response_img;
            LUT(images[i], log_response, response_img);
            split(response_img, splitted);
            for(int c = 0; c < channels; c++) {
                result_split[c] += w.mul(splitted[c] - times[(int)i]);
            }
            weight_sum += w;
        }
        weight_sum = 1.0f / weight_sum;
        for(int c = 0; c < channels; c++) {
            result_split[c] = result_split[c].mul(weight_sum);
        }
        merge(result_split, result);
        exp(result, result);
        cv::FileStorage file("result.ext", cv::FileStorage::WRITE);
        file << "result" << result;
    }    

protected:
    String name;
    Mat weights;
};

void process(std::vector<py::array_t<unsigned char>> src, py::array_t<float> times, py::array_t<float> resDebevec){
    std::vector<Mat> mat_src = images_pytocpp(src);
    std::vector<float> time_vec = times_pytocpp(times);
    Mat resDebevec_mt = debs_pytocpp(resDebevec);
    Merge obj = Merge();
    obj.process(mat_src, time_vec, resDebevec_mt);    
}
PYBIND11_MODULE(_merge, m) {
	m.doc() = "_merge";    
     m.def("process", &process, "process");
}