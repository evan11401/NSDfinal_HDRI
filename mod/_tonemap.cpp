#include "tool.cpp"

inline void log_(const Mat& src, Mat& dst)
{
    max(src, Scalar::all(1e-4), dst);
    log(dst, dst);
}

class TonemapGamma
{
public:
    TonemapGamma(float _gamma=1.0f) : name("Tonemap"), gamma(_gamma)
    {
    }

    Mat process(Mat src)  
    {
        Mat dst;

        double min, max;
        minMaxLoc(src, &min, &max);
        if(max - min > DBL_EPSILON) {
            dst = (src - min) / (max - min);
        } else {
            src.copyTo(dst);
        }

        pow(dst, 1.0f / gamma, dst);
        cv::FileStorage file("TonemapGamma.ext", cv::FileStorage::WRITE);
        file << "result" << dst;
        return dst;
    }

    float getGamma() const   { return gamma; }
    void setGamma(float val)   { gamma = val; }
    
    void read(const FileNode& fn)  
    {
        FileNode n = fn["name"];
        gamma = fn["gamma"];
    }

protected:
    String name;
    float gamma;
};


class TonemapDrag
{
public:
    TonemapDrag(float _gamma=1.0f, float _saturation=0.7f, float _bias=0.85f) :
        name("TonemapDrag"),
        gamma(_gamma),
        saturation(_saturation),
        bias(_bias)
    {
    }

    void process(Mat src)  
    {
        Mat img;

        TonemapGamma linear = TonemapGamma(1.0f);
        img = linear.process(src);

        Mat gray_img;
        cvtColor(img, gray_img, COLOR_RGB2GRAY);
        Mat log_img;
        log_(gray_img, log_img);
        float mean = expf(static_cast<float>(sum(log_img)[0]) / log_img.total());
        gray_img /= mean;
        log_img.release();

        double max;
        minMaxLoc(gray_img, NULL, &max);

        Mat map;
        log(gray_img + 1.0f, map);
        Mat div;
        pow(gray_img / static_cast<float>(max), logf(bias) / logf(0.5f), div);
        log(2.0f + 8.0f * div, div);
        map = map.mul(1.0f / div);
        div.release();

        mapLuminance(img, img, gray_img, map, saturation);

        linear.setGamma(gamma);
        img = linear.process(img);
        cv::FileStorage file("TonemapDrag.ext", cv::FileStorage::WRITE);
        file << "result" << img;
    }

    float getGamma() const   { return gamma; }
    void setGamma(float val)   { gamma = val; }

    float getSaturation() const   { return saturation; }
    void setSaturation(float val)   { saturation = val; }

    float getBias() const   { return bias; }
    void setBias(float val)   { bias = val; }   

    void read(const FileNode& fn)  
    {
        FileNode n = fn["name"];
        gamma = fn["gamma"];
        bias = fn["bias"];
        saturation = fn["saturation"];
    }

protected:
    String name;
    float gamma, saturation, bias;
};
void process(py::array_t<float> hdrDebevec){
    Mat hdr_mat = debs_pytocpp(hdrDebevec);
    TonemapGamma objGamma = TonemapGamma();
    objGamma.setGamma(3);
    objGamma.process(hdr_mat);
}
void processDrag(py::array_t<float> hdrDebevec){
    Mat hdr_mat = debs_pytocpp(hdrDebevec);
    TonemapDrag objGrago = TonemapDrag();
    objGrago.process(hdr_mat);
}
PYBIND11_MODULE(_tonemap, m) {
	m.doc() = "_tonemap";    
    m.def("process", &process, "process");
    m.def("processDrag", &processDrag, "processDrag");
}