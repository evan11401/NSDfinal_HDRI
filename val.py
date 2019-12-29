#!/bin/bash

''':'
test_path="${BASH_SOURCE[0]}"

fail_msg="*** validation failed"
source /opt/intel/bin/compilervars.sh intel64

# SKIP_CLEAN will not be set during grading.
if [ -z "$SKIP_CLEAN" ] ; then
    make clean ; ret=$?
    if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

    if [ -n "$(ls _align_image*.so 2> /dev/null)" ] ; then
        echo "$fail_msg for uncleanness"
        exit 1
    fi
fi

make ; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

python3 -m pytest $test_path -v -s ; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "validation pass"

exit 0
':'''

import unittest
import timeit
import os

import cv2
import numpy as np
import scipy.io as sio


import _align
import _calCRF
import _merge
import _tonemap

def readImagesAndTimes():
  
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  
  filenames = ["imgs/img_0.033.jpg", "imgs/img_0.25.jpg", "imgs/img_2.5.jpg", "imgs/img_15.jpg"]
  
  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times

images, times = readImagesAndTimes()
one = np.array([1]*(256*1*3), dtype=np.float32)
resDebevec = None
hdrDebevec = None
class GradingTest(unittest.TestCase):
     
    def test_align(self):
        global images
        images_cv = images.copy()
        images_our = images.copy()
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images_cv)        
        images_our = _align.process(images)
        for i in range(len(images)):
            self.assertEqual(np.array_equal(images_cv[i], images_our[i]), True)
            self.assertEqual(images_cv[i].shape, images_our[i].shape)
        images = images_our

    def test_calCRF(self):
        global images, times, resDebevec
        time_cv = times.copy()
        time_our = times.copy()
        print("Calculating Camera Response Function (CRF) ... ")
        calibrateDebevec = cv2.createCalibrateDebevec()
        resDebevec_cv = calibrateDebevec.process(images, time_cv)
        resDebevec = resDebevec_cv
        resDebevec_our = _calCRF.process(images, time_our)
        resDebevec_our_np = np.array(resDebevec_our, dtype=np.float32)
        resDebevec_our_np.resize((256, 1, 3))
        twoten_np = np.array([20]*(256*1*3), dtype=np.float32)
        twoten_np.resize((256, 1, 3))
        sub_np = np.absolute(np.array(resDebevec_our_np) - np.array(resDebevec_our))
        self.assertGreater(twoten_np.all(), sub_np.all())
        

    def test_merge(self):
        global images, times, resDebevec, hdrDebevec, one
        print("Merging images into one HDR image ... ")
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(images, times, resDebevec)
        _merge.process(images, times, resDebevec)
        cv_file = cv2.FileStorage("result.ext", cv2.FILE_STORAGE_READ)
        hdrDebevec_our = cv_file.getNode("result").mat()
        sub_np = np.absolute(np.subtract(hdrDebevec_our, hdrDebevec))
        
        self.assertGreater(one.all(), sub_np.all())
        
    def test_tonemap(self):
        global hdrDebevec, one
        print("Tonemaping using Gamma and Drago's method ... ")
        tonemapGamma = cv2.createTonemap(3)
        ldrGamma = tonemapGamma.process(hdrDebevec)
        _tonemap.process(hdrDebevec)
        cv_file = cv2.FileStorage("TonemapGamma.ext", cv2.FILE_STORAGE_READ)
        ldrGamma_our = cv_file.getNode("result").mat()
        sub_np = np.absolute(np.subtract(ldrGamma_our, ldrGamma))
        self.assertGreater(one.all(), sub_np.all())
        hdrDebevec = 3 * hdrDebevec
        cv2.imwrite("hdrDebevec.jpg", hdrDebevec * 255)
        print("saved hdrDebevec.jpg")
        ldrGamma = 3 * ldrGamma
        cv2.imwrite("ldrGamma_cv.jpg", ldrGamma * 255)
        print("saved ldrGamma_cv.jpg")
        ldrGamma_our = 3 * ldrGamma_our
        cv2.imwrite("ldrGamma_our.jpg", ldrGamma_our * 255)
        print("saved ldrGamma_our.jpg")


        

# vim: set fenc=utf8 ff=unix et sw=4 ts=4 sts=4: