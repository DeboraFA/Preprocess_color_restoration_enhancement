import numpy as np
from PIL import Image
import click
import os


from deblurgan.model import generator_model
from deblurgan.utils import load_image, deprocess_image, preprocess_image

weight_path = 'generator.h5'
def deblur(path_dir):
    g = generator_model()
    g.load_weights(weight_path)
    
    image = np.array([preprocess_image(load_image(path_dir))])
    x_test = image
    generated_images = g.predict(x=x_test)
    generated = np.array(deprocess_image(generated_images))
#     x_test = deprocess_image(x_test)
#     x = x_test[:, :, :]
    img = generated[:, :, :]
#     print(np.shape(x))
#     print(np.shape(img))
#     output = np.concatenate((x, img), axis=1)
#     im = Image.fromarray(img)
    return img[-1]



