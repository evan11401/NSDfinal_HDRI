#include "env.hpp"
#include"converter.h"


class Align
{
public:
    Align(int max_bits = 6, int exclude_range = 4, bool cut = true) :
        name("AlignMTB"),
        max_bits(max_bits),
        exclude_range(exclude_range),
        cut(cut)
    {
    }

    /*void process(InputArrayOfArrays src, std::vector<Mat>& dst,
                 InputArray, InputArray) CV_OVERRIDE
    {
        process(src, dst);
    }*/
    
    void process(std::vector<Mat>& src, std::vector<Mat>& dst)
    {

        //std::vector<Mat> src;
        //_src.getMatVector(src);

        dst.resize(src.size());

        size_t pivot = src.size() / 2;
        dst[pivot] = src[pivot];
        Mat gray_base;
        cvtColor(src[pivot], gray_base, COLOR_RGB2GRAY);
        std::vector<Point> shifts;

        for(size_t i = 0; i < src.size(); i++) {
            if(i == pivot) {
                shifts.push_back(Point(0, 0));
                continue;
            }
            Mat gray;
            cvtColor(src[i], gray, COLOR_RGB2GRAY);
            Point shift = calculateShift(gray_base, gray);
            shifts.push_back(shift);
            shiftMat(src[i], dst[i], shift);
        }
        if(cut) {
            Point max(0, 0), min(0, 0);
            for(size_t i = 0; i < shifts.size(); i++) {
                if(shifts[i].x > max.x) {
                    max.x = shifts[i].x;
                }
                if(shifts[i].y > max.y) {
                    max.y = shifts[i].y;
                }
                if(shifts[i].x < min.x) {
                    min.x = shifts[i].x;
                }
                if(shifts[i].y < min.y) {
                    min.y = shifts[i].y;
                }
            }
            Point size = dst[0].size();
            for(size_t i = 0; i < dst.size(); i++) {
                dst[i] = dst[i](Rect(max, min + size));
            }
        }
    }

    Point calculateShift(InputArray _img0, InputArray _img1) 
    {

        Mat img0 = _img0.getMat();
        Mat img1 = _img1.getMat();
        CV_Assert(img0.channels() == 1 && img0.type() == img1.type());
        CV_Assert(img0.size() == img1.size());

        int maxlevel = static_cast<int>(log((double)max(img0.rows, img0.cols)) / log(2.0)) - 1;
        maxlevel = min(maxlevel, max_bits - 1);

        std::vector<Mat> pyr0;
        std::vector<Mat> pyr1;
        buildPyr(img0, pyr0, maxlevel);
        buildPyr(img1, pyr1, maxlevel);

        Point shift(0, 0);
        for(int level = maxlevel; level >= 0; level--) {

            shift *= 2;
            Mat tb1, tb2, eb1, eb2;
            computeBitmaps(pyr0[level], tb1, eb1);
            computeBitmaps(pyr1[level], tb2, eb2);

            int min_err = (int)pyr0[level].total();
            Point new_shift(shift);
            for(int i = -1; i <= 1; i++) {
                for(int j = -1; j <= 1; j++) {
                    Point test_shift = shift + Point(i, j);
                    Mat shifted_tb2, shifted_eb2, diff;
                    shiftMat(tb2, shifted_tb2, test_shift);
                    shiftMat(eb2, shifted_eb2, test_shift);
                    bitwise_xor(tb1, shifted_tb2, diff);
                    bitwise_and(diff, eb1, diff);
                    bitwise_and(diff, shifted_eb2, diff);
                    int err = countNonZero(diff);
                    if(err < min_err) {
                        new_shift = test_shift;
                        min_err = err;
                    }
                }
            }
            shift = new_shift;
        }
        return shift;
    }

    void shiftMat(InputArray _src, OutputArray _dst, const Point shift) 
    {
        Mat src = _src.getMat();
        _dst.create(src.size(), src.type());
        Mat dst = _dst.getMat();

        Mat res = Mat::zeros(src.size(), src.type());
        int width = src.cols - abs(shift.x);
        int height = src.rows - abs(shift.y);
        Rect dst_rect(max(shift.x, 0), max(shift.y, 0), width, height);
        Rect src_rect(max(-shift.x, 0), max(-shift.y, 0), width, height);
        src(src_rect).copyTo(res(dst_rect));
        res.copyTo(dst);
    }

    int getMaxBits() const  { return max_bits; }
    void setMaxBits(int val)  { max_bits = val; }

    int getExcludeRange() const  { return exclude_range; }
    void setExcludeRange(int val)  { exclude_range = val; }

    bool getCut() const  { return cut; }
    void setCut(bool val)  { cut = val; }

    void read(const FileNode& fn) 
    {
        FileNode n = fn["name"];
        CV_Assert(n.isString() && String(n) == name);
        max_bits = fn["max_bits"];
        exclude_range = fn["exclude_range"];
        int cut_val = fn["cut"];
        cut = (cut_val != 0);
    }

    void computeBitmaps(InputArray _img, OutputArray _tb, OutputArray _eb) 
    {
        Mat img = _img.getMat();
        _tb.create(img.size(), CV_8U);
        _eb.create(img.size(), CV_8U);
        Mat tb = _tb.getMat(), eb = _eb.getMat();
        int median = getMedian(img);
        compare(img, median, tb, CMP_GT);
        compare(abs(img - median), exclude_range, eb, CMP_GT);
    }

protected:
    String name;
    int max_bits, exclude_range;
    bool cut;

    void downsample(Mat& src, Mat& dst)
    {
        dst = Mat(src.rows / 2, src.cols / 2, CV_8UC1);

        int offset = src.cols * 2;
        uchar *src_ptr = src.ptr();
        uchar *dst_ptr = dst.ptr();
        for(int y = 0; y < dst.rows; y ++) {
            uchar *ptr = src_ptr;
            for(int x = 0; x < dst.cols; x++) {
                dst_ptr[0] = ptr[0];
                dst_ptr++;
                ptr += 2;
            }
            src_ptr += offset;
        }
    }

    void buildPyr(Mat& img, std::vector<Mat>& pyr, int maxlevel)
    {
        pyr.resize(maxlevel + 1);
        pyr[0] = img.clone();
        for(int level = 0; level < maxlevel; level++) {
            downsample(pyr[level], pyr[level + 1]);
        }
    }

    int getMedian(Mat& img)
    {
        int channels = 0;
        Mat hist;
        int hist_size = LDR_SIZE;
        float range[] = {0, LDR_SIZE} ;
        const float* ranges[] = {range};
        calcHist(&img, 1, &channels, Mat(), hist, 1, &hist_size, ranges);
        float *ptr = hist.ptr<float>();
        int median = 0, sum = 0;
        int thresh = (int)img.total() / 2;
        while(sum < thresh && median < LDR_SIZE) {
            sum += static_cast<int>(ptr[median]);
            median++;
        }
        return median;
    }
};

py::array copy_array(py::array in_arr){
    py::array ret_arr = py::array(in_arr.dtype(),
                                  {in_arr.shape(), in_arr.shape()+in_arr.ndim()},
                                  {in_arr.strides(), in_arr.strides()+in_arr.ndim()},
                                  in_arr.data(),
                                  in_arr);
    return ret_arr;
}
std::vector<py::array_t<unsigned char>> process(std::vector<py::array_t<unsigned char>> src){
    std::vector<py::array_t<unsigned char>> ret_vector;
    ret_vector.assign(src.begin(), src.end());
    Align obj = Align();
    //obj.process(src, ret_vector);
    vector<Mat> mat_src;
    for(uint i=0;i<src.size();i++){
        py::buffer_info buf = src[i].request();
        Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
        imwrite("aaa.jpg", mat);
    }
    return ret_vector;
}


PYBIND11_MODULE(_align, m) {
	m.doc() = "Align";    
    //m.def("process", &processs);
    m.def("copy_arr", &copy_array, "copy array (generic)");
    m.def("process", &process, "process");
   
}