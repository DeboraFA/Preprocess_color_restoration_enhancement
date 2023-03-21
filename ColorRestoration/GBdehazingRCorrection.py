from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 
import PIL
import os
import datetime
import numpy as np
import cv2
import natsort
from DetermineDepth import determineDepth
from TransmissionEstimation import getTransmission
from getAdaptiveExposureMap import AdaptiveExposureMap
from getAdaptiveSceneRadiance import AdaptiveSceneRadiance
from getAtomsphericLight import getAtomsphericLight
from refinedTransmission import refinedtransmission

from sceneRadianceGb import sceneRadianceGB
from sceneRadianceR import sceneradiance

# # # # # # # # # # # # # # # # # # # # # # Normalized implement is necessary part as the fore-processing   # # # # # # # # # # # # # # # #


np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def GBdehazingRC(path): 
    img = Image.open(path)
    img = img_to_array(img)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    blockSize = 9
    largestDiff = determineDepth(img, blockSize)
    AtomsphericLight, AtomsphericLightGB, AtomsphericLightRGB = getAtomsphericLight(largestDiff, img)
#         print('AtomsphericLightRGB',AtomsphericLightRGB)
    
    # transmission = getTransmission(img, AtomsphericLightRGB, blockSize=blockSize)
    transmission = getTransmission(img, AtomsphericLightRGB, blockSize)
    # print('transmission.shape',transmission.shape)
    # TransmissionComposition(folder, transmission, number, param='coarse')
    transmission = refinedtransmission(transmission, img)

    # TransmissionComposition(folder, transmission, number, param='refined_15_175_175')
    sceneRadiance_GB = sceneRadianceGB(img, transmission, AtomsphericLightRGB)

    
    # # print('sceneRadiance_GB',sceneRadiance_GB)
    sceneRadiance = sceneradiance(img, sceneRadiance_GB)

    S_x = AdaptiveExposureMap(img, sceneRadiance, Lambda=0.3, blockSize=blockSize)
    sceneRadiance = AdaptiveSceneRadiance(sceneRadiance, S_x)
    
    return sceneRadiance
        


