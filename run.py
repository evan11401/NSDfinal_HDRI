import unittest
import timeit
import os
import sys
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

if __name__ == '__main__':
    print("Reading images and times ... ")
    images, times = readImagesAndTimes()
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
            
    print("Calculating Camera Response Function (CRF) ... ")
    resDebevec = _calCRF.process(images, times)
    resDebevec = np.array(resDebevec, dtype=np.float32)
    resDebevec.resize((256, 1, 3)) 

    print("Merging images into one HDR image ... ")
    _merge.process(images, times, resDebevec)
    cv_file = cv2.FileStorage("result.ext", cv2.FILE_STORAGE_READ)
    hdrDebevec = cv_file.getNode("result").mat()
    
    # Save HDR image.
    cv2.imwrite("hdrDebevec_our.jpg", hdrDebevec)
    print("saved hdrDebevec.jpg ")
    

    print("Tonemaping using Gamma and Drago's method ... ")
    _tonemap.process(hdrDebevec)
    cv_file = cv2.FileStorage("TonemapGamma.ext", cv2.FILE_STORAGE_READ)
    ldrGamma = cv_file.getNode("result").mat()
    ldrGamma = 3 * ldrGamma
    cv2.imwrite("ldrGamma.jpg", ldrGamma * 255)
    print("saved ldrGamma.jpg")

    _tonemap.processDrag(hdrDebevec)
    cv_file = cv2.FileStorage("TonemapDrag.ext", cv2.FILE_STORAGE_READ)
    ldrDrag = cv_file.getNode("result").mat()
    ldrDrag = 3 * ldrDrag
    cv2.imwrite("ldrDrag.jpg", ldrDrag * 255)
    print("saved ldrDrag.jpg")