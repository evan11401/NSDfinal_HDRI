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

import _align

def readImagesAndTimes():
  
  times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  
  filenames = ["imgs/img_0.033.jpg", "imgs/img_0.25.jpg", "imgs/img_2.5.jpg", "imgs/img_15.jpg"]
  
  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images, times


class GradingTest(unittest.TestCase):

    def test_align(self):
        images, times = readImagesAndTimes()
        images_cv = images.copy()
        print(images[0].shape)
        # Align input images
        print("Aligning images ... ")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images_cv)
        
        images_our = _align.process(images)
        # images_our = _align.copy_arr(images[0])
        #for i in range(len(images)):
            #self.assertEqual(images_cv[i].shape, images_our[i].shape)


# vim: set fenc=utf8 ff=unix et sw=4 ts=4 sts=4: