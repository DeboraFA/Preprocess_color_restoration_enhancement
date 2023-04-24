# std
import argparse
from argparse import RawTextHelpFormatter
import glob
from os import makedirs
from os.path import join, exists, basename, splitext
# 3p
import cv2
from tqdm import tqdm
# project
from exposure_enhancement import enhance_image_exposure


def LLIE_main(file):
    # load images
    gamma = 0.6
    lambda_ = 0.15
    sigma = 3
    bc = 1
    bs = 1
    be = 1
    eps = 1e-3
    
    
    ext = ['png', 'jpg', 'bmp']    # Add image formats here
   
    image = cv2.imread(file)

    # enhance images
    enhanced_image = enhance_image_exposure(image, gamma, lambda_,
                                            sigma=sigma, bc=bc, bs=bs, be=be, eps=eps)
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    return enhanced_image



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Python implementation of two low-light image enhancement techniques via illumination map estimation.",
#         formatter_class=RawTextHelpFormatter
#     )
#     parser.add_argument("-f", '--folder', default='./demo/', type=str,
#                         help="folder path to test images.")
#     parser.add_argument("-g", '--gamma', default=0.6, type=float,
#                         help="the gamma correction parameter.")
#     parser.add_argument("-l", '--lambda_', default=0.15, type=float,
#                         help="the weight for balancing the two terms in the illumination refinement optimization objective.")
#     parser.add_argument("-ul", "--lime", action='store_true',
#                         help="Use the LIME method. By default, the DUAL method is used.")
#     parser.add_argument("-s", '--sigma', default=3, type=int,
#                         help="Spatial standard deviation for spatial affinity based Gaussian weights.")
#     parser.add_argument("-bc", default=1, type=float,
#                         help="parameter for controlling the influence of Mertens's contrast measure.")
#     parser.add_argument("-bs", default=1, type=float,
#                         help="parameter for controlling the influence of Mertens's saturation measure.")
#     parser.add_argument("-be", default=1, type=float,
#                         help="parameter for controlling the influence of Mertens's well exposedness measure.")
#     parser.add_argument("-eps", default=1e-3, type=float,
#                         help="constant to avoid computation instability.")

#     args = parser.parse_args()
#     main(args)
