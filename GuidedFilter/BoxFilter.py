import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class BoxFilter(tf.keras.layers.Layer):
	def __init__(self, radious=1, eps=1e-8, nhwc=True, **kwargs):
		super().__init__(**kwargs)
		self.radious = radious
		self.eps     = eps
		self.nhwc    = nhwc

	def __diff_x__(self, input, r):
		assert input.shape.ndims == 4
		left   = input[:, :,     r:2*r +1]
		middle = input[:, :, 2*r+1:      ] -input[:, :,       :-2*r-1]
		right  = input[:, :,    -1:      ] -input[:, :, -2*r-1:  -r-1]
		output = tf.concat([left, middle, right], axis=2)
		return output
				
	def __diff_y__(self, input, r):
		assert input.shape.ndims == 4
		left   = input[:, :, :,     r:2*r+1]
		middle = input[:, :, :, 2*r+1:     ] -input[:, :, :,       :-2*r-1]
		right  = input[:, :, :,    -1:     ] -input[:, :, :, -2*r-1:  -r-1]
		output = tf.concat([left, middle, right], axis=3)
		return output

	def box_filter(self, x, r):
		assert x.shape.ndims == 4
		return self.__diff_y__(tf.cumsum(self.__diff_x__(tf.cumsum(x, axis=2), r), axis=3), r)

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
