
import os
import numpy as np
import cv2
import natsort
import xlwt
from skimage import exposure

from sceneRadianceCLAHE import RecoverCLAHE
from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def HE(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    sceneRadiance = RecoverHE(img)
    return sceneRadiance
