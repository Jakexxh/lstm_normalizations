import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class Vani_Norm_BasicRNNCell(RNNCell):
	"""The most basic RNN cell.

Args:
  num_units: int, The number of units in the RNN cell.
  activation: Nonlinearity to use.  Default: `tanh`.
  reuse: (optional) Python boolean describing whether to reuse variables
   in an existing scope.  If not `True`, and the existing scope already has
   the given variables, an error is raised.
  name: String, the name of the layer. Layers with the same name will
	share weights, but to avoid mistakes we require reuse=True in such
	cases.
"""

	def __init__(self,
	             num_units,
	             mode,
	             is_training_tensor,
	             max_steps=-1,
	             activation=None,
	             reuse=None,
	             name=None):
		super(Vani_Norm_BasicRNNCell, self).__init__(_reuse=reuse, name=name)

		self._num_units = num_units
		self._mode = mode
		self._training = is_training_tensor
		self._max_steps = max_steps
		self._activation = activation or tf.tanh

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def build(self, inputs_shape):
		if inputs_shape[1].value is None:
			raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
			                 % inputs_shape)

		input_depth = inputs_shape[1].value
		self._kernel = self.add_variable(
			_WEIGHTS_VARIABLE_NAME,
			shape=[input_depth + self._num_units, self._num_units])
		self._bias = self.add_variable(
			_BIAS_VARIABLE_NAME,
			shape=[self._num_units],
			initializer=tf.zeros_initializer(dtype=self.dtype))

		self.built = True

	def call(self, inputs, state):
		"""Most basic RNN: output = new_state = act(W * input + U * state + B)."""
		input_depth = tf.shape(inputs)[1]
		if self._mode == 'base':
			gate_inputs = tf.matmul(tf.concat([inputs, state], 1), self._kernel)
			gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
			output = self._activation(gate_inputs)
			return output, output

		elif self._mode == 'bn_sep':
			h, step = state
			_step = tf.squeeze(tf.gather(tf.cast(step, tf.int32), 0))
			xh_gate = self._batch_norm(tf.matmul(inputs, self._kernel[:input_depth, :]), 'bn_xh', _step)
			hh_gate = self._batch_norm(tf.matmul(state, self._kernel[input_depth:, :]), 'bn_hh', _step)
			gate_inputs = xh_gate + hh_gate
			output = self._activation(gate_inputs)
			return output, (output, step + 1)

		elif self._mode == 'cn_sep':
			xh_gate = self._cosine_norm(inputs, self._kernel[:input_depth, :], 'cn_xh')
			hh_gate = self._cosine_norm(state, self._kernel[input_depth:, :], 'cn_hh')
			gate_inputs = xh_gate + hh_gate
			gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
			output = self._activation(gate_inputs)
			return output, output

		elif self._mode == 'pcc_sep':
			xh_gate = self._pcc_norm(inputs, self._kernel[:input_depth, :], 'pcc_xh')
			hh_gate = self._pcc_norm(state, self._kernel[input_depth:, :], 'pcc_hh')
			gate_inputs = xh_gate + hh_gate
			gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
			output = self._activation(gate_inputs)
			return output, output

		elif self._mode == 'ln_sep':
			xh_gate = self._layer_norm(tf.matmul(inputs, self._kernel[:input_depth, :]), scope='ln_xh')
			hh_gate = self._layer_norm(tf.matmul(state, self._kernel[input_depth:, :]), scope='ln_hh')
			gate_inputs = xh_gate + hh_gate
			gate_inputs = self._layer_norm(gate_inputs)
			output = self._activation(gate_inputs)
			return output, output

		elif self._mode == 'wn_sep':
			xh_gate = self._weight_norm(inputs, self._kernel[:input_depth, :], 'wn_xh')
			hh_gate = self._weight_norm(state, self._kernel[input_depth:, :], 'wn_hh')
			gate_inputs = xh_gate + hh_gate
			gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
			output = self._activation(gate_inputs)
			return output, output
		else:
			raise Exception("Invalid mode!")

	def _batch_norm(self,
	                x,
	                name_scope,
	                step,
	                initial_scale=0.1,
	                decay=0.95,
	                epsilon=1e-7,
	                no_offset=False,
	                set_forget_gate_bias=False):
		'''Assume 2d [batch, values] tensor'''

		with tf.variable_scope(name_scope):
			size = x.get_shape().as_list()[1]

			scale = tf.get_variable(
				'scale', [size],
				initializer=tf.truncated_normal_initializer(
					initial_scale))
			if no_offset:
				offset = 0
			elif set_forget_gate_bias:
				offset = tf.get_variable(
					'offset', [size], initializer=offset_initializer())
			else:
				offset = tf.get_variable(
					'offset', [size], initializer=tf.zeros_initializer)

			pop_mean_all_steps = tf.get_variable(
				'pop_mean_all', [self._max_steps, size],
				initializer=tf.zeros_initializer,
				trainable=False)
			pop_var_all_steps = tf.get_variable(
				'pop_var_all', [self._max_steps, size],
				initializer=tf.ones_initializer(),
				trainable=False)

			step = tf.minimum(step, self._max_steps - 1)

			pop_mean = pop_mean_all_steps[step]
			pop_var = pop_var_all_steps[step]

			batch_mean, batch_var = tf.nn.moments(x, [0])

			def batch_statistics():
				pop_mean_new = pop_mean * decay + batch_mean * (
					1 - decay)
				pop_var_new = pop_var * decay + batch_var * (
					1 - decay)
				with tf.control_dependencies([
					pop_mean.assign(pop_mean_new),
					pop_var.assign(pop_var_new)
				]):
					return tf.nn.batch_normalization(x, batch_mean, batch_var,
					                                 offset, scale, epsilon)

			def population_statistics():
				return tf.nn.batch_normalization(x, pop_mean, pop_var, offset,
				                                 scale, epsilon)

			if type(self._training) == bool:
				if self._training:
					return batch_statistics()
				else:
					return population_statistics()
			else:
				return tf.cond(self._training, batch_statistics,
				               population_statistics)

	def _cosine_norm(self, x, w, name='cosine_norm'):
		with tf.name_scope(name):
			x = tf.concat([x, tf.fill([tf.shape(x)[0], 1], 1e-7)], axis=1)

			w = tf.concat([w, tf.fill([1, tf.shape(w)[1]], 1e-7)], axis=0)

			if tf.equal(tf.shape(x)[1], tf.shape(w)[0]) is not None:

				x_l2 = tf.nn.l2_normalize(x, 1)

				w_l2 = tf.nn.l2_normalize(w, 0)

				cos_mat = tf.matmul(x_l2, w_l2)
				gamma = tf.get_variable(
					name + '_gamma', [cos_mat.get_shape().as_list()[1]],
					initializer=tf.truncated_normal_initializer(1.0))

				return gamma * cos_mat

			else:
				raise Exception(
					'Matrix shape does not match in cosine_norm Operation!')

	def _layer_norm(self, inputs, epsilon=1e-7, scope=None):
		# TODO: may be optimized
		mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
		with tf.variable_scope(scope + 'ln'):
			scale = tf.get_variable(
				'alpha',
				shape=[inputs.get_shape()[1]],
				initializer=tf.truncated_normal_initializer(0.1))
			shift = tf.get_variable(
				'beta',
				shape=[inputs.get_shape()[1]],
				initializer=tf.zeros_initializer)
		res = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

		return res

	def _pcc_norm(self, x, w, name=None):
		with tf.name_scope(name + '_pcc_norm'):
			x = tf.concat([x, tf.fill([tf.shape(x)[0], 1], 1e-7)], axis=1)

			w = tf.concat([w, tf.fill([1, tf.shape(w)[1]], 1e-7)], axis=0)

			x_mean, _ = tf.nn.moments(x, [1], keep_dims=True)
			w_mean, _ = tf.nn.moments(w, [0], keep_dims=True)
			if tf.equal(tf.shape(x)[1], tf.shape(w)[0]) is not None:

				x_l2 = tf.nn.l2_normalize(x - x_mean, 1)

				w_l2 = tf.nn.l2_normalize(w - w_mean, 0)

				cos_mat = tf.matmul(x_l2, w_l2)

				gamma = tf.get_variable(
					name + '_gamma', [cos_mat.get_shape().as_list()[1]],
					initializer=tf.truncated_normal_initializer(1.0))

				return gamma * cos_mat
			else:
				raise Exception(
					'Matrix shape does not match in cosine_norm Operation!')

	def _weight_norm(self, x, V, scope='weight_norm'):
		with tf.name_scope(scope):
			shape = V.get_shape().as_list()
			g = tf.get_variable(
				name=scope + '_g',
				shape=[
					shape[1],
				],
				dtype=tf.float32,
				initializer=tf.truncated_normal_initializer(1.0))

			w = g * tf.nn.l2_normalize(V, 0)

		return tf.matmul(x, w)


def offset_initializer():
	def _initializer(shape, dtype=tf.float32, partition_info=None):
		size = shape[0]
		assert size % 4 == 0
		size = size // 4
		res = [np.ones((size)), np.zeros((size * 3))]
		return tf.constant(np.concatenate(res, axis=0), dtype)

	return _initializer
