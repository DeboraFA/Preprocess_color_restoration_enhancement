import math
import os
import natsort
import numpy as np
import datetime
import cv2
from skimage.color import rgb2hsv


from color_equalisation import RGB_equalisation
from global_stretching_RGB import stretching
from hsvStretching import HSVStretching2 as HSVStretching

from histogramDistributionLower import histogramStretching_Lower
from histogramDistributionUpper import histogramStretching_Upper
from rayleighDistribution import rayleighStretching
from rayleighDistributionLower import rayleighStretching_Lower
from rayleighDistributionUpper import rayleighStretching_Upper
from sceneRadiance import sceneRadianceRGB

e = np.e
esp = 2.2204e-16
np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def Rayleigh(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = len(img)
    width = len(img[0])

    sceneRadiance = RGB_equalisation(img, height, width)
    # sceneRadiance = stretching(img)
    sceneRadiance = stretching(sceneRadiance)
    sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(sceneRadiance, height, width)

    sceneRadiance = (np.float64(sceneRadiance_Lower) + np.float64(sceneRadiance_Upper)) / 2

    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
 
    return sceneRadiance