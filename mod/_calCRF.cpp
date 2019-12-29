#include "env.hpp"


class calCRF
{
public:
    calCRF(int _samples=70, float _lambda=10.0f, bool _random=false) :
        name("CalibrateDebevec"),
        samples(_samples),
        lambda(_lambda),
        random(_random),
        w(triangleWeights())
    {
    }

    std::vector<Mat> process(std::vector<Mat> images, Mat dst, std::vector<float> times) 
    {
        int channels = images[0].channels();
        int CV_32FCC = CV_MAKETYPE(CV_32F, channels);
        int rows = images[0].rows;
        int cols = images[0].cols;

        dst.create(LDR_SIZE, 1, CV_32FCC);
        Mat result = dst;

        // pick pixel locations (either random or in a rectangular grid)
        std::vector<Point> points;
        points.reserve(samples);
        if(random) {
            for(int i = 0; i < samples; i++) {
                points.push_back(Point(rand() % cols, rand() % rows));
            }
        } else {
            int x_points = static_cast<int>(sqrt(static_cast<double>(samples) * cols / rows));
            int y_points = samples / x_points;
            int step_x = cols / x_points;
            int step_y = rows / y_points;
            for(int i = 0, x = step_x / 2; i < x_points; i++, x += step_x) {
                for(int j = 0, y = step_y / 2; j < y_points; j++, y += step_y) {
                    if( 0 <= x && x < cols && 0 <= y && y < rows ) {
                        points.push_back(Point(x, y));
                    }
                }
            }
            // we can have slightly less grid points than specified
            //samples = static_cast<int>(points.size());
        }
        
        // we need enough equations to ensure a sufficiently overdetermined system
        // (maybe only as a warning)
        //CV_Assert(points.size() * (images.size() - 1) >= LDR_SIZE);

        // solve for imaging system response function, over each channel separately
        std::vector<Mat> result_split(channels);
        
        for(int ch = 0; ch < channels; ch++) {
            // initialize system of linear equations
            Mat A = Mat::zeros((int)points.size() * (int)images.size() + LDR_SIZE + 1,
                LDR_SIZE + (int)points.size(), CV_32F);
            Mat B = Mat::zeros(A.rows, 1, CV_32F);

            // include the data-fitting equations
            int k = 0;
            for(size_t i = 0; i < points.size(); i++) {
                for(size_t j = 0; j < images.size(); j++) {
                    int val = images[j].ptr()[channels*(points[i].y * cols + points[i].x) + ch];
                    float wij = w.at<float>(val);
                    A.at<float>(k, val) = wij;
                    A.at<float>(k, LDR_SIZE + (int)i) = -wij;
                    B.at<float>(k, 0) = wij * log(times[(int)j]);
                    k++;
                }
            }

            // fix the curve by setting its middle value to 0
            A.at<float>(k, LDR_SIZE / 2) = 1;
            k++;

            // include the smoothness equations
            for(int i = 0; i < (LDR_SIZE - 2); i++) {
                float wi = w.at<float>(i + 1);
                A.at<float>(k, i) = lambda * wi;
                A.at<float>(k, i + 1) = -2 * lambda * wi;
                A.at<float>(k, i + 2) = lambda * wi;
                k++;
            }

            // solve the overdetermined system using SVD (least-squares problem)
            Mat solution;
            solve(A, B, solution, DECOMP_SVD);
            solution.rowRange(0, LDR_SIZE).copyTo(result_split[ch]);
        }
        // combine log-exposures and take its exponent
        for(int i=0;i<3;i++)
            exp(result_split[i], result_split[i]);
        return result_split;
        merge(result_split, result);
        exp(result, result);
    }

    int getSamples() const  { return samples; }
    void setSamples(int val)  { samples = val; }

    float getLambda() const  { return lambda; }
    void setLambda(float val)  { lambda = val; }

    bool getRandom() const  { return random; }
    void setRandom(bool val)  { random = val; }    

    void read(const FileNode& fn) 
    {
        FileNode n = fn["name"];
        samples = fn["samples"];
        lambda = fn["lambda"];
        int random_val = fn["random"];
        random = (random_val != 0);
    }

protected:
    String name;  // calibration algorithm identifier
    int samples;  // number of pixel locations to sample
    float lambda; // constant that determines the amount of smoothness
    bool random;  // whether to sample locations randomly or in a grid shape
    Mat w;        // weighting function for corresponding pixel values
};

std::vector<std::vector<float>> process(std::vector<py::array_t<unsigned char>> src, py::array_t<float> times){
    
    calCRF obj = calCRF();
    std::vector<Mat> mat_src = images_pytocpp(src);
    std::vector<float> time_vec = times_pytocpp(times);
    Mat mat_ret ;
    
    std::vector<Mat> ret_spi = obj.process(mat_src, mat_ret, time_vec);
    std::vector<std::vector<float>> ret_vector;
    ret_vector.resize(256);
    for(int i=0;i<256;i++){
        ret_vector[i].resize(3);
    }
    for(int i=0;i<3;i++){
        Mat mat = ret_spi[i];
        std::vector<float> dst;
        if (mat.isContinuous()) {
            dst.assign((float*)mat.data, (float*)mat.data + mat.total());
        } else {
            for (int i = 0; i < mat.rows; ++i) {
                dst.insert(dst.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols);
            }
        }
        for(int j=0;j<256;j++){
            ret_vector[j][i] = dst[j];
        }
    }    
    return ret_vector;
}

PYBIND11_MODULE(_calCRF, m) {
	m.doc() = "_calCRF";    
     m.def("process", &process, "process");
   
}