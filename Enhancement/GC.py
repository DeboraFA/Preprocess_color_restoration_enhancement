
import os
import numpy as np
import cv2
import natsort
import xlwt
from skimage import exposure

from sceneRadianceGC import RecoverGC

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def GC(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sceneRadiance = RecoverGC(img)
    return sceneRadiance
    