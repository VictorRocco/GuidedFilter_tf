#!/usr/bin/env python3

print("""
Fast Guided Filter Self Smoothing.
A scaled Image is used as filtering guide to smooth itself preserving edges.
""")

#--- CONFIGURATION ---
INPUT_RGB_PATH = 'bicycle_rgb_input.jpg'
INPUT_MASK_PATH= 'bicycle_mask_input.png'
RESIZED_IMAGE_SHAPE= 512

#--- IMPORTS ---
import sys, os, cv2
import numpy as np
import auxiliary_functions as aux

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Avoid TF warnings.
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from GuidedFilter import FastGuidedFilter

#--- MAIN PROGRAM ---

#hr_rgb = cv2.imread(INPUT_RGB_PATH, cv2.IMREAD_GRAYSCALE) #High resolution Gray image
hr_rgb = cv2.imread(INPUT_RGB_PATH) #High resolution RGB image
if hr_rgb is None:
	print("ERROR: reading image", INPUT_RGB_PATH)
	print("Hint: check that the image is inside the directory that you are running this test.")
	exit(1)

print("Input image:", INPUT_MASK_PATH, "- size:", hr_rgb.shape[0], "x", hr_rgb.shape[1], 
	"- Mpx: {:0.2f}".format(hr_rgb.shape[0] * hr_rgb.shape[1] / 1024/1024)) 

lr_rgb = cv2.resize(hr_rgb, (RESIZED_IMAGE_SHAPE, RESIZED_IMAGE_SHAPE), cv2.INTER_AREA) #Low resolution conversion
scaled_lr_rgb = cv2.resize(hr_rgb, (int(RESIZED_IMAGE_SHAPE/4.0), int(RESIZED_IMAGE_SHAPE/4.0)), cv2.INTER_AREA) #Low resolution conversion

print("LR image:", INPUT_MASK_PATH, "- size:", lr_rgb.shape[0], "x", lr_rgb.shape[1], 
	"- Mpx: {:0.2f}".format(lr_rgb.shape[0] * lr_rgb.shape[1] / 1024/1024)) 

lr_rgb_tf = aux.image_to_normalized_tensor(lr_rgb) #Conversion to Tensorflow input
scaled_lr_rgb_tf = aux.image_to_normalized_tensor(scaled_lr_rgb) #Conversion to Tensorflow input

GF = FastGuidedFilter()

GF.set_config(radious=int(16), eps=1.0/100.0, nhwc=True)
lr_rgb_tf_guided_filtered = GF(scaled_lr_rgb_tf, scaled_lr_rgb_tf, lr_rgb_tf) #guiding image, guided image
lr_rgb_tf_guided_filtered = GF(scaled_lr_rgb_tf, scaled_lr_rgb_tf, lr_rgb_tf_guided_filtered) #guiding image, guided image
lr_rgb_tf_guided_filtered = GF(scaled_lr_rgb_tf, scaled_lr_rgb_tf, lr_rgb_tf_guided_filtered) #guiding image, guided image

psnr = tf.image.psnr(lr_rgb_tf, lr_rgb_tf_guided_filtered, max_val=1.0)
tf.print("PSNR:", psnr)

lr_rgb_guided_filtered = aux.normalized_tensor_to_image(lr_rgb_tf_guided_filtered)

h_image = cv2.hconcat((lr_rgb, lr_rgb_guided_filtered))

cv2.imshow("Q quit, N Next", h_image)

aux.wait_for_key() #Q quit, N Next
