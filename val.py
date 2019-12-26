#!/bin/bash

''':'
test_path="${BASH_SOURCE[0]}"

if [[ (-n "$PRELOAD_MKL") && ("Linux" == "$(uname)") ]] ; then
    # Workaround for cmake + MKL in conda.
    MKL_ROOT=$HOME/opt/conda
    MKL_LIB_DIR=$MKL_ROOT/lib
    MKL_LIBS=$MKL_LIB_DIR/libmkl_def.so
    MKL_LIBS=$MKL_LIBS:$MKL_LIB_DIR/libmkl_avx2.so
    MKL_LIBS=$MKL_LIBS:$MKL_LIB_DIR/libmkl_core.so
    MKL_LIBS=$MKL_LIBS:$MKL_LIB_DIR/libmkl_intel_lp64.so
    MKL_LIBS=$MKL_LIBS:$MKL_LIB_DIR/libmkl_sequential.so
    export LD_PRELOAD=$MKL_LIBS
    echo "set LD_PRELOAD=$LD_PRELOAD for MKL"
else
    echo "set PRELOAD_MKL if you see (Linux) MKL linking error"
fi

fail_msg="*** validation failed"

# SKIP_CLEAN will not be set during grading.
if [ -z "$SKIP_CLEAN" ] ; then
    make clean ; ret=$?
    if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

    if [ -n "$(ls _matrix*.so 2> /dev/null)" ] ; then
        echo "$fail_msg for uncleanness"
        exit 1
    fi
    echo "GET POINT 1"
fi

make ; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi
if [ -z "$SKIP_CLEAN" ] ; then
    echo "GET POINT 1"
fi

python3 -m pytest $test_path -v -s ; ret=$?
if [ 0 -ne $ret ] ; then echo "$fail_msg" ; exit $ret ; fi

echo "validation pass"
if [ -z "$SKIP_CLEAN" ] ; then
    echo "GET POINT 3"
fi
exit 0
':'''

import unittest
import timeit
import os

import cv2
import numpy as np

# The python module that wraps the matrix code.
import _align

def readImagesAndTimes():
  
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  
  filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
  
  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times


class GradingTest(unittest.TestCase):

    def test_basic(self):
        images, times = readImagesAndTimes()
        # Align input images
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images_origin)
        
        _align.process(images, images_our)
        self.assertEqual(images_origin.shape, images_our.shape)
        

# vim: set fenc=utf8 ff=unix et sw=4 ts=4 sts=4: