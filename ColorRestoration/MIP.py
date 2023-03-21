import os
import numpy as np
import cv2
import natsort

from BL import getAtomsphericLight
from EstimateDepth import DepthMap
from getRefinedTramsmission import Refinedtransmission
from TM import getTransmission
from sceneRadiance import sceneRadianceRGB2

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def MIP(path):
    img = cv2.imread(path)

    blockSize = 9

    largestDiff = DepthMap(img, blockSize)
    transmission = getTransmission(largestDiff)
    transmission = Refinedtransmission(transmission,img)
    AtomsphericLight = getAtomsphericLight(transmission, img)
    sceneRadiance = sceneRadianceRGB2(img, transmission, AtomsphericLight)
    return sceneRadiance