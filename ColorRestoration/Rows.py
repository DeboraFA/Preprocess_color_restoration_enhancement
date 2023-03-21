import os
import numpy as np
import cv2
import natsort
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 

from refinedTransmission import Refinedtransmission2 as Refinedtransmission
from getAtomsphericLight import getAtomsphericLight3 as getAtomsphericLight
from getRGBDarkChannel import getRGBDarkchannel as getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB2 as sceneRadianceRGB

from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def Rows(path):
    img = Image.open(path)
    img = img_to_array(img)     
    blockSize = 9

    RGB_Darkchannel = getDarkChannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight(RGB_Darkchannel, img)
    #         print('AtomsphericLight', AtomsphericLight)
    transmission = getTransmission(img, AtomsphericLight, blockSize)
    #         print('transmission',transmission)
    #         print('np.mean(transmission)',np.mean(transmission))
    transmission = Refinedtransmission(transmission, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
    return sceneRadiance



