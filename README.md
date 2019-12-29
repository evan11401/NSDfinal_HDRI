# NSDfinal_HDRI
NCTUCS NSD final project-High Dynamic Range Imaging using Multiple Exposure Images with Tone Mapping

# Install
### make sure opencv c++ installed

### to create pybind objects
```
make clean
make
```
### to run whole code
```
python run.py
```
### to run the test
```
./val.py
```

### to run in other images
```
#at run.py line 13,14
times = np.array([ Your Shutter's Times ], dtype=np.float32)
filenames = [Your Image Path]
```
### then run
```
python run.py
```

# Reference

## Image Alignment
#### cv::AlignMTB
### Greg Ward. [Fast, robust image registration for compositing high dynamic range photographs from hand-held exposures.](http://www.anyhere.com/gward/papers/jgtpap2.pdf) Journal of graphics tools, 8(2):17–30, 2003.


## Construct HDR
#### cv::CalibrateDebevec
#### cv::MergeDebevec
### Paul E Debevec and Jitendra Malik. [Recovering high dynamic range radiance maps from photographs.](http://www.pauldebevec.com/Research/HDR/debevec-siggraph97.pdf) In ACM SIGGRAPH 2008 classes, page 31. ACM, 2008.

## Tone mapping
#### cv::TonemapDrago
### Frédéric Drago, Karol Myszkowski, Thomas Annen, and Norishige Chiba. [Adaptive logarithmic mapping for displaying high contrast scenes.](http://resources.mpi-inf.mpg.de/tmo/logmap/logmap.pdf) In Computer Graphics Forum, volume 22, pages 419–426. Wiley, 2003.
