import tensorflow as tf
import numpy as np
# import time
from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.platform import tf_logging as logging
from .__init__ import weights_initializer

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class CNLSTMCell(RNNCell):
    def __init__(self,
                 num_units,
                 grain,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None):

        super(CNLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
        self._grain = grain
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        sigmoid = math_ops.sigmoid
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
        i, j, f, o = self._line_sep([inputs, h], self._num_units, bias=False)


        new_c = (c * sigmoid(f + self._forget_bias) +
                 sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state

    def _line_sep(self,
                  args,
                  h_size,
                  bias,
                  bias_initializer=None,
                  kernel_initializer=None):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError("linear is expecting 2D arguments: %s" % shapes)
            if shape[1].value is None:
                raise ValueError("linear expects shape[1] to \
                                 be provided for shape %s, "
                                 "but saw %s" % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as outer_scope:

            [x, h] = args

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable(
                'W_xh', [x_size, h_size * 4], initializer=weights_initializer
            )
            W_ih = tf.get_variable(
                'W_ih', [h_size, h_size], initializer=weights_initializer
            )
            W_jh = tf.get_variable(
                'W_jh', [h_size, h_size], initializer=weights_initializer
            )
            W_fh = tf.get_variable(
                'W_fh', [h_size, h_size], initializer=weights_initializer
            )
            W_oh = tf.get_variable(
                'W_oh', [h_size, h_size], initializer=weights_initializer
            )

            cn_xh = self.cosine_norm(x, W_xh, 'cn_xh')

            cn_ih = self.cosine_norm(h, W_ih, 'cn_ih') + cn_xh[:, :h_size]
            cn_jh = self.cosine_norm(h, W_jh, 'cn_jh') + cn_xh[:, h_size:h_size * 2]
            cn_fh = self.cosine_norm(h, W_fh, 'cn_fh') + cn_xh[:, h_size * 2:h_size * 3]
            cn_oh = self.cosine_norm(h, W_oh, 'cn_oh') + cn_xh[:, h_size * 3:]

            if not bias:
                return cn_ih, cn_jh, cn_fh, cn_oh
            with vs.variable_scope(outer_scope) as inner_scope:
                inner_scope.set_partitioner(None)
                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(
                        0.0, dtype=dtype)
                biases_i = vs.get_variable(
                    _BIAS_VARIABLE_NAME + 'i', [h_size],
                    dtype=dtype,
                    initializer=bias_initializer)
                biases_j = vs.get_variable(
                    _BIAS_VARIABLE_NAME + 'j', [h_size],
                    dtype=dtype,
                    initializer=bias_initializer)
                biases_o = vs.get_variable(
                    _BIAS_VARIABLE_NAME + 'o', [h_size],
                    dtype=dtype,
                    initializer=bias_initializer)
                biases_f = vs.get_variable(
                    _BIAS_VARIABLE_NAME + 'f', [h_size],
                    dtype=dtype,
                    initializer=bias_initializer)

            return (nn_ops.bias_add(cn_ih, biases_i), nn_ops.bias_add(
                cn_jh, biases_j), nn_ops.bias_add(cn_oh, biases_o),
                    nn_ops.bias_add(cn_fh, biases_f))

    def cosine_norm(self, x, w, name='cosine_norm'):
        with tf.name_scope(name):
            x = tf.concat([x, tf.fill([tf.shape(x)[0], 1], 1e-7)], axis=1)

            w = tf.concat([w, tf.fill([1, tf.shape(w)[1]], 1e-7)], axis=0)

            if tf.equal(tf.shape(x)[1], tf.shape(w)[0]) is not None:

                x_l2 = tf.nn.l2_normalize(x, 1)

                w_l2 = tf.nn.l2_normalize(w, 0)

                cos_mat = tf.matmul(x_l2, w_l2)
                gamma = tf.get_variable(
                	name + '_gamma', [cos_mat.get_shape().as_list()[1]],
                    	initializer=tf.truncated_normal_initializer(self._grain))
                beta = tf.get_variable(
               		name + '_beta', [cos_mat.get_shape().as_list()[1]],
                	initializer=tf.zeros_initializer)

                return gamma * cos_mat + beta

            else:
                raise Exception(
                    'Matrix shape does not match in cosine_norm Operation!')


def identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        # gate (j) is identity
        # t = np.zeros(shape)
        t = np.identity(size) * scale
        return tf.constant(t, dtype)

    return _initializer
