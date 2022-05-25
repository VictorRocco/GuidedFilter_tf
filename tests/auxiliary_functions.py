
import tensorflow as tf
import numpy as np
import cv2, sys

#Convert cv2 image to normalized TF tensor
def image_to_normalized_tensor(image):
	if len(image.shape) > 2:
		if image.shape[2] == 4: #PNG 4 channels -> JPG 3 channels
			image = image[:, :, :3] 
	normalized_image = (image / 255.0).astype(np.float32) #normalized range 0-1, float dtype
	normalized_tensor = tf.constant(normalized_image, dtype=tf.float32) #to tensor
	
	#RGB (height, width, channels) or Gray (height, width) -> (1, height, width, channels) 
	normalized_tensor = tf.expand_dims(normalized_tensor, axis=0)
	if normalized_tensor.shape.ndims == 3:
		normalized_tensor = tf.expand_dims(normalized_tensor, axis=-1)
			
	#tf.print("image_to_normalized_tensor:", normalized_tensor.shape)
	return normalized_tensor

#Convert normalized TF tensor to cv2 image
def normalized_tensor_to_image(normalized_tensor):
	numpy = np.asarray(normalized_tensor * 255) #range 0-1 -> 0-255
	numpy = numpy.clip(0, 255+1) #range sanity check
	image = numpy.squeeze() #(1, height, width, channels) -> (height, width, channels) or (height, width)
	image = image.astype(np.uint8)
	return image


def wait_for_key():
	while True:
		ch = cv2.waitKey(1)
		if ch == 27 or ch == ord('q') or ch == ord('Q'):
			sys.exit()
			break
		if ch == ord('n') or ch == ord('N'):
			break
	cv2.destroyAllWindows()
