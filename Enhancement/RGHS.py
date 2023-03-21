
# encoding=utf-8
import os
import numpy as np
import cv2
import natsort

from LabStretching import LABStretching
from color_equalisation import RGB_equalisation2 as RGB_equalisation
from global_stretching_RGB import stretching
from relativeglobalhistogramstretching import RelativeGHstretching

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def RGHS(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = len(img)
    width = len(img[0])

    sceneRadiance = img

    sceneRadiance = stretching(sceneRadiance)


    sceneRadiance = LABStretching(sceneRadiance)

    return sceneRadiance