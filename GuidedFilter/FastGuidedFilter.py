import tensorflow as tf
from .BoxFilter import BoxFilter

@tf.keras.utils.register_keras_serializable()
class FastGuidedFilter(BoxFilter):
	def __init__(self, radious=1, eps=1e-8, nhwc=True, **kwargs):
		super(FastGuidedFilter, self).__init__(radious, eps, nhwc, **kwargs)
		self.radious = radious
		self.eps     = eps
		self.nhwc    = nhwc

	def fast_guided_filter(self, guiding_image_low_resolution, guided_image_low_resolution, guided_image_high_resolution, 
						radious, eps, nhwc):
		lr_x = guiding_image_low_resolution
		lr_y = guided_image_low_resolution
		hr_x = guided_image_high_resolution
		self.radious = radious
		self.eps = eps
		self.nhwc = nhwc

		# shape check
		assert lr_x.shape.ndims == 4
		assert lr_y.shape.ndims == 4 
		assert hr_x.shape.ndims == 4

		# data format
		if nhwc:
			lr_x = tf.transpose(lr_x, [0, 3, 1, 2])
			lr_y = tf.transpose(lr_y, [0, 3, 1, 2])
			hr_x = tf.transpose(hr_x, [0, 3, 1, 2])

		# shape check
		lr_x_shape = tf.shape(lr_x)
		lr_y_shape = tf.shape(lr_y)
		hr_x_shape = tf.shape(hr_x)

		# shape check
		tf.assert_equal(lr_x_shape[0], lr_y_shape[0])
		tf.assert_equal(lr_x_shape[0], hr_x_shape[0])
		tf.assert_equal(lr_x_shape[1], hr_x_shape[1])
		tf.assert_equal(lr_x_shape[2:], lr_y_shape[2:])
		tf.assert_greater(lr_x_shape[2:], 2 * self.radious + 1)
		tf.Assert(tf.logical_or(tf.equal(lr_x_shape[1], 1), tf.equal(lr_x_shape[1], lr_y_shape[1])), [lr_x_shape, lr_y_shape])

		# compute
		# N
		N = self.box_filter(tf.ones((1, 1, lr_x_shape[2], lr_x_shape[3]), dtype=lr_x.dtype), self.radious)
		# mean_x
		mean_x = self.box_filter(lr_x, self.radious) / N
		# mean_y
		mean_y = self.box_filter(lr_y, self.radious) / N
		# cov_xy
		cov_xy = self.box_filter(lr_x * lr_y, self.radious) / N - mean_x * mean_y
		# var_xself.
		var_x  = self.box_filter(lr_x * lr_x, self.radious) / N - mean_x * mean_x
		# A
		A = cov_xy / (var_x + self.eps)
		# b
		b = mean_y - A * mean_x
		# mean_A; mean_b
		A    = tf.transpose(A,    [0, 2, 3, 1])
		b    = tf.transpose(b,    [0, 2, 3, 1])
		hr_x = tf.transpose(hr_x, [0, 2, 3, 1])
		mean_A = tf.image.resize(A, hr_x_shape[2:])
		mean_b = tf.image.resize(b, hr_x_shape[2:])
		#output = mean_A * hr_x + mean_b
		output = mean_A * tf.dtypes.cast(hr_x, tf.float32) + mean_b

		# data format
		if not nhwc:
			output = tf.transpose(output, [0, 3, 1, 2])

		return output

	def call(self, guiding_image_low_resolution, guided_image_low_resolution, guided_image_high_resolution, 
			radious=None, eps=None, nhwc=None):

		self.radios = self.radious if radious==None else radious
		self.eps = self.eps if eps==None else eps
		self.nhwc = self.nhwc if nhwc==None else nhwc

		return self.fast_guided_filter(guiding_image_low_resolution, guided_image_low_resolution, guided_image_high_resolution,
				self.radious, self.eps, self.nhwc)

	def set_config(self, radious=1, eps=1e-8, nhwc=True):
		self.radious = radious
		self.eps     = eps
		self.nhwc    = nhwc

	def get_config(self):
		config = super().get_config()
		config["radious"] = self.radious
		config["eps"] = self.eps
		config["nhwc"] = self.nhwc
		return config