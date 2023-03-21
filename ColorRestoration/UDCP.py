import os
import numpy as np
import cv2
import natsort

from refinedTransmission import Refinedtransmission2 as Refinedtransmission
from getAtomsphericLight import getAtomsphericLight3 as getAtomsphericLight
from getRGBDarkChannel import getRGBDarkchannel as getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB2 as sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

def UDCP(path):
    img = cv2.imread(path)

    blockSize = 9
    GB_Darkchannel = getDarkChannel(img, blockSize)
    AtomsphericLight = getAtomsphericLight(GB_Darkchannel, img)

    # print('AtomsphericLight', AtomsphericLight)
    # print('img/AtomsphericLight', img/AtomsphericLight)

    # AtomsphericLight = [231, 171, 60]

    transmission = getTransmission(img, AtomsphericLight, blockSize)

    # cv2.imwrite('OutputImages/' + prefix + '_UDCP_Map.jpg', np.uint8(transmission))

    transmission = Refinedtransmission(transmission, img)
    sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight)
    return sceneRadiance


