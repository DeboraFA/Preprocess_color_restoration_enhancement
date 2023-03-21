from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 
import os
import numpy as np
import cv2
import natsort


from DetermineDepth import determineDepth
from getAtomsphericLight import getAtomsphericLight3 as getAtomsphericLight
from getRefinedTramsmission import Refinedtransmission2 as Refinedtransmission
from getScatteringRate import ScatteringRateMap
from getSceneRadiance import SceneRadiance2 as SceneRadiance
from getTransmissionGB import TransmissionGB
from getTransmissionR import TransmissionR2 as TransmissionR

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def NewOpticalModel(path):
    img = Image.open(path)
    img = img_to_array(img)
    blockSize = 9
    largestDiff = determineDepth(img, blockSize)
    AtomsphericLight = getAtomsphericLight(largestDiff, img)
    # print('AtomsphericLight',AtomsphericLight)
    sactterRate = ScatteringRateMap(img, AtomsphericLight, blockSize)
    # print('sactterRate',sactterRate)
    transmissionGB = TransmissionGB(sactterRate)
    transmissionR = TransmissionR(transmissionGB, img, blockSize)
    transmissionGB, transmissionR = Refinedtransmission(transmissionGB, transmissionR, img)
    sceneRadiance = SceneRadiance(img, transmissionGB, transmissionR, sactterRate, AtomsphericLight)
    return sceneRadiance
