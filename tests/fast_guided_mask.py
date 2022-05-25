#!/usr/bin/env python3

#--- CONFIGURATION ---
INPUT_RGB_PATH = 'bicycle_rgb_input.jpg'
INPUT_MASK_PATH= 'bicycle_mask_input.png'
RESIZED_IMAGE_SHAPE= 512

#--- IMPORTS ---
import sys, os, cv2
import numpy as np
import auxiliary_functions as aux

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from GuidedFilters import FastGuidedFilter

#--- MAIN PROGRAM ---

input_rgb = cv2.imread(INPUT_RGB_PATH)
input_mask= cv2.imread(INPUT_MASK_PATH)
	
print("Input image:", INPUT_MASK_PATH, "- size:", input_rgb.shape[0], "x", input_rgb.shape[1], 
	"- Mpx: {:0.2f}".format(input_rgb.shape[0] * input_rgb.shape[1] / 1024/1024)) 
print("Mask image :", INPUT_MASK_PATH, "- size:", input_mask.shape[0], "x", input_mask.shape[1], 
	"- Mpx: {:0.2f}".format(input_mask.shape[0] * input_mask.shape[1] /1024/1024)) 

lr_rgb = cv2.resize(input_rgb, (RESIZED_IMAGE_SHAPE, RESIZED_IMAGE_SHAPE), cv2.INTER_AREA)
lr_mask= cv2.resize(input_mask, (RESIZED_IMAGE_SHAPE, RESIZED_IMAGE_SHAPE), cv2.INTER_AREA)

print("LR image:", INPUT_MASK_PATH, "- size:", lr_rgb.shape[0], "x", lr_rgb.shape[1], 
	"- Mpx: {:0.2f}".format(lr_rgb.shape[0] * lr_rgb.shape[1] / 1024/1024)) 
print("LR image :", INPUT_MASK_PATH, "- size:", lr_mask.shape[0], "x", lr_mask.shape[1], 
	"- Mpx: {:0.2f}".format(lr_mask.shape[0] * lr_mask.shape[1] /1024/1024)) 


hr_rgb_tf = aux.image_to_normalized_tensor(input_rgb)
hr_mask_tf= aux.image_to_normalized_tensor(input_mask)
lr_rgb_tf = aux.image_to_normalized_tensor(lr_rgb)
lr_mask_tf= aux.image_to_normalized_tensor(lr_mask)

FGF = FastGuidedFilter()
	
gt1 = tfa.image.median_filter2d(gt, (int(MIN_SIDE/32.0), int(MIN_SIDE/32.0)))
small_gt1  = tf.image.resize(gt1,  small_shape)


FGF.set_config(radious=int(MIN_SIDE/64.0), eps=1.0/1000.0, nhwc=True)
gt2 = FGF(small_rgb, small_gt1, gt1, radious=int(MIN_SIDE/64.0), eps=1.0/1000.0, nhwc=True) #realzo bordes

FGF.set_config(radious=int(MIN_SIDE/32.0), eps=1.0/100.0, nhwc=True)
rgbgt_1 = FGF(small_rgb, small_gt, gt)
gtrgb_1 = FGF(small_gt, small_rgb, rgb)

rgbgt_1 = np.asarray(rgbgt_1.clip(0, 1) * 255, dtype=np.uint8)
rgbgt_1 = rgbgt_1.squeeze()
rgbgt_1 = cv2.resize(rgbgt_1, (PREDICT_OUTPUT_IMAGE_SHAPE, PREDICT_OUTPUT_IMAGE_SHAPE))

gtrgb_1 = np.asarray(gtrgb_1.clip(0, 1) * 255, dtype=np.uint8)
gtrgb_1 = gtrgb_1.squeeze()
gtrgb_1 = cv2.resize(gtrgb_1, (PREDICT_OUTPUT_IMAGE_SHAPE, PREDICT_OUTPUT_IMAGE_SHAPE))

from skimage import img_as_ubyte
image1 = img_as_ubyte(image1)

image1 = cv2.resize(image1, (PREDICT_OUTPUT_IMAGE_SHAPE, PREDICT_OUTPUT_IMAGE_SHAPE), cv2.INTER_CUBIC)

mask1 = np.asarray(gt2.clip(0, 1) * 255, dtype=np.uint8)
mask1 = mask1.squeeze()
mask1 = cv2.resize(mask1, (PREDICT_OUTPUT_IMAGE_SHAPE, PREDICT_OUTPUT_IMAGE_SHAPE), cv2.INTER_CUBIC)

print("mask1:", mask1.shape, mask1.dtype)
print("rgbgt_1:", rgbgt_1.shape, rgbgt_1.dtype)
print("image1:", image1.shape, image1.dtype)
print("gtrgb_1:", gtrgb_1.shape, gtrgb_1.dtype)

h1_image = cv2.hconcat((mask1, rgbgt_1))
h2_image = cv2.hconcat((image1, gtrgb_1))
v_image = cv2.vconcat((h1_image, h2_image))

cv2.imshow("Q quit, N Next", v_image)

while True:
	ch = cv2.waitKey(1)
	if ch == 27 or ch == ord('q') or ch == ord('Q'):
		sys.exit()
		break
	if ch == ord('n') or ch == ord('N'):
		break
cv2.destroyAllWindows()
