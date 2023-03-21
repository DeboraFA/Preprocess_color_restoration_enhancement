import os
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image 
import datetime
import numpy as np
import cv2
import natsort
from CloseDepth import closePoint
from F_stretching import StretchingFusion
from MapFusion import Scene_depth
from MapOne import max_R
from MapTwo import R_minus_GB
from blurrinessMap import blurrnessMap
from getAtomsphericLightFusion import ThreeAtomsphericLightFusion
from getAtomsphericLightOne import getAtomsphericLightDCP_Bright
from getAtomsphericLightThree import getAtomsphericLightLb
from getAtomsphericLightTwo import getAtomsphericLightLv
from getRGBDarkChannel import getRGBDarkchannel
from getRefinedTransmission import Refinedtransmission
from getTransmissionGB import getGBTransmissionESt
from getTransmissionR import getTransmission
from global_Stretching import global_stretching
from sceneRadiance import sceneRadianceRGB
from sceneRadianceHE import RecoverHE

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def IBLA(path):
    img = Image.open(path)
    img = img_to_array(img)
        
    blockSize = 9
    n = 5
    RGBDarkchannel = getRGBDarkchannel(img, blockSize)
    BlurrnessMap = blurrnessMap(img, blockSize, n)
    AtomsphericLightOne = getAtomsphericLightDCP_Bright(RGBDarkchannel, img, percent=0.001)
    AtomsphericLightTwo = getAtomsphericLightLv(img)
    AtomsphericLightThree = getAtomsphericLightLb(img, blockSize, n)
    AtomsphericLight = ThreeAtomsphericLightFusion(AtomsphericLightOne, AtomsphericLightTwo, AtomsphericLightThree, img)
    print('AtomsphericLight',AtomsphericLight)   # [b,g,r]


    R_map = max_R(img, blockSize)
    mip_map = R_minus_GB(img, blockSize, R_map)
    bluriness_map = BlurrnessMap

    d_R = 1 - StretchingFusion(R_map)
    d_D = 1 - StretchingFusion(mip_map)
    d_B = 1 - StretchingFusion(bluriness_map)

    d_n = Scene_depth(d_R, d_D, d_B, img, AtomsphericLight)
    d_n_stretching = global_stretching(d_n)
    d_0 = closePoint(img, AtomsphericLight)
    d_f = 8  * (d_n +  d_0)

    transmissionR = getTransmission(d_f)
    transmissionB, transmissionG = getGBTransmissionESt(transmissionR, AtomsphericLight)
    transmissionB, transmissionG, transmissionR = Refinedtransmission(transmissionB, transmissionG, transmissionR, img)

    sceneRadiance = sceneRadianceRGB(img, transmissionB, transmissionG, transmissionR, AtomsphericLight)
    
    return sceneRadiance


