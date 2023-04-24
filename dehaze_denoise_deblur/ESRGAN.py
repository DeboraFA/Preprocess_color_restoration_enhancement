#!/usr/bin/env python
# coding: utf-8

# # Image Super Resolution using ESRGAN

# In[1]:


import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
# pip install tensorflow_hub ==0.11.0
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)


# In[18]:


def esrgan_main(IMAGE_PATH):
    hr_image = preprocess_image(IMAGE_PATH)
    model = hub.load(SAVED_MODEL_PATH)
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    image = tf.squeeze(fake_image)
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    return image

