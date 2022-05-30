#!/usr/bin/env python3

print("""
Guided Filter Mask Smoothing.
An Image is used as filtering guide to smooth the Foreground/Background mask edges.
Used in Deep Learning Image Segmentation to refine borders.
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

from GuidedFilter import GuidedFilter

#--- MAIN PROGRAM ---

hr_rgb = cv2.imread(INPUT_RGB_PATH) # High resolution RGB image
#hr_rgb = cv2.imread(INPUT_RGB_PATH, cv2.IMREAD_GRAYSCALE) # High resolution Gray image
if hr_rgb is None:
	print("ERROR: reading image", INPUT_RGB_PATH)
	print("Hint: check that the image is inside the directory that you are running this test.")
	exit(1)
hr_mask = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_COLOR) # High resolution RGB image
#hr_mask = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_GRAYSCALE) # High resolution Gray image
if hr_mask is None:
	print("ERROR: reading image", INPUT_MASK_PATH)
	print("Hint: check that the image is inside the directory that you are running this test.")
	exit(1)

print("Input image:", INPUT_RGB_PATH, "- size:", hr_rgb.shape[0], "x", hr_rgb.shape[1],
	"- Mpx: {:0.2f}".format(hr_rgb.shape[0] * hr_rgb.shape[1] / 1024/1024)) 
print("Mask image:", INPUT_MASK_PATH, "- size:", hr_mask.shape[0], "x", hr_mask.shape[1],
	"- Mpx: {:0.2f}".format(hr_mask.shape[0] * hr_mask.shape[1] / 1024/1024))

# low resolution conversion
lr_rgb = cv2.resize(hr_rgb, (RESIZED_IMAGE_SHAPE, RESIZED_IMAGE_SHAPE), cv2.INTER_AREA)
lr_mask = cv2.resize(hr_mask, (RESIZED_IMAGE_SHAPE, RESIZED_IMAGE_SHAPE), cv2.INTER_AREA)

print("LR image:", INPUT_RGB_PATH, "- size:", lr_rgb.shape[0], "x", lr_rgb.shape[1],
	"- Mpx: {:0.2f}".format(lr_rgb.shape[0] * lr_rgb.shape[1] / 1024/1024)) 
print("LR mask:", INPUT_MASK_PATH, "- size:", lr_mask.shape[0], "x", lr_mask.shape[1],
	"- Mpx: {:0.2f}".format(lr_mask.shape[0] * lr_mask.shape[1] / 1024/1024))

# Dilation / Erosion of mask
kernel = np.ones((5, 5), np.uint8)
modified_lr_mask = cv2.dilate(lr_mask, kernel, iterations=1)
modified_lr_mask = cv2.erode(modified_lr_mask, kernel, iterations=1)
# modified_lr_mask = cv2.erode(lr_mask, kernel, iterations=1)
# modified_lr_mask = cv2.dilate(modified_lr_mask, kernel, iterations=1)
# modified_lr_mask = lr_mask

# Conversion to Tensorflow input
lr_rgb_tf = aux.image_to_normalized_tensor(lr_rgb)
lr_mask_tf = aux.image_to_normalized_tensor(lr_mask)
modified_lr_mask_tf = aux.image_to_normalized_tensor(modified_lr_mask)

GF = GuidedFilter()
GF.set_config(radious=int(1), eps=5.0/10.0, nhwc=True)

lr_mask_tf_guided_filtered = GF(lr_rgb_tf, modified_lr_mask_tf) # guiding image, guided image
#lr_mask_tf_guided_filtered = GF(modified_lr_mask_tf, lr_rgb_tf) # guiding image, guided image

psnr = tf.image.psnr(lr_mask_tf, lr_mask_tf_guided_filtered, max_val=1.0)
tf.print("PSNR:", psnr)

lr_mask_guided_filtered = aux.normalized_tensor_to_image(lr_mask_tf_guided_filtered)

h_image = cv2.hconcat((modified_lr_mask, lr_mask_guided_filtered))

cv2.imshow("Q quit, N Next", h_image)

aux.wait_for_key() # Q quit, N Next
