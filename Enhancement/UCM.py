import os
import numpy as np
import cv2
import natsort
import xlwt
import datetime

from color_equalisation import RGB_equalisation2 as RGB_equalisation
from global_histogram_stretching import stretching2 as stretching
from hsvStretching import HSVStretching
from sceneRadiance import sceneRadianceRGB


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def UCM(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print('Number',Number)
    sceneRadiance = RGB_equalisation(img)
    sceneRadiance = stretching(sceneRadiance)
    # # cv2.imwrite(folder + '/OutputImages/' + Number + 'Stretched.jpg', sceneRadiance)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance
