import os
import numpy as np
import cv2
import natsort
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 

from TransmissionMap import TransmissionComposition
from getAtomsphericLight import getAtomsphericLight, getAtomsphericLight2
from getColorContrastEnhancement import ColorContrastEnhancement
from getRGBDarkChannel import getRGBDarkchannel
from getSceneRadiance import SceneRadiance





######################## Based on the DCP and the 0.1% brightest point is incorrect ########################
######################## Based on the DCP and the 0.1% brightest point is incorrect ########################
######################## Based on the DCP and the 0.1% brightest point is incorrect  and further cause the distortion of the restored images ########################
from getTransmissionEstimation import getTransmissionMap

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def LowComplexityDCP(path):
    
    img = Image.open(path)
    img = img_to_array(img)
    blockSize = 9

    imgGray = getRGBDarkchannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight2(imgGray, img, percent=0.001)
    # print('AtomsphericLight',AtomsphericLight)
    transmission = getTransmissionMap(img, AtomsphericLight, blockSize)
    sceneRadiance = SceneRadiance(img, AtomsphericLight, transmission)
    sceneRadiance = ColorContrastEnhancement(sceneRadiance)
    return sceneRadiance
