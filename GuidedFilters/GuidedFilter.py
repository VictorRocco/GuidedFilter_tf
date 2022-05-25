import tensorflow as tf
from .GuidedFilterBase import GuidedFilterBase

class GuidedFilter(GuidedFilterBase):
	def __init__(self, radious=1, eps=1e-8, nhwc=True):
		super().__init__(radious, eps, nhwc)
		
	def guided_filter(self, guiding_image, guided_image, radious, eps, nhwc):
		x = guiding_image
		y = guided_image
		self.radious = radious
		self.eps = eps
		self.nhwc = nhwc
		assert x.shape.ndims == 4 
		assert y.shape.ndims == 4
		# data format
		if nhwc:
			x = tf.transpose(a=x, perm=[0, 3, 1, 2])
			y = tf.transpose(a=y, perm=[0, 3, 1, 2])
		# shape check
		x_shape = tf.shape(input=x)
		y_shape = tf.shape(input=y)
		assets = [tf.assert_equal(    x_shape[0],  y_shape[0]),
					tf.assert_equal(  x_shape[2:], y_shape[2:]),
					tf.assert_greater(x_shape[2:], 2*self.radious+1),
					tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
						tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]
		with tf.control_dependencies(assets):
			x = tf.identity(x)
		#function
		# N
		N = self.__box_filter__(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), self.radious)
		# mean_x
		mean_x = self.__box_filter__(x, self.radious) / N
		# mean y
		mean_y = self.__box_filter__(y, self.radious) / N
		# cov_xy
		cov_xy = self.__box_filter__(x * y, self.radious) / N - mean_x * mean_y
		# var_x
		var_x  = self.__box_filter__(x * x, self.radious) / N - mean_x * mean_x
		# A
		A = cov_xy / (var_x + self.eps)
		# b
		b = mean_y - A * mean_x
		mean_A = self.__box_filter__(A, self.radious) / N
		mean_b = self.__box_filter__(b, self.radious) / N
		output = mean_A * x + mean_b
		if nhwc:
			output = tf.transpose(a=output, perm=[0, 2, 3, 1])
		return output
		
	def call(self, guiding_image, guided_image, radious=None, eps=None, nhwc=None):
		self.radios = self.radious if radious==None else radious
		self.eps = self.eps if eps==None else eps
		self.nhwc = self.nhwc if nhwc==None else nhwc
		return self.guided_filter(guiding_image, guided_image, self.radious, self.eps, self.nhwc)
			

