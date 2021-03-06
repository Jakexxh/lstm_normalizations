import numpy as np
import tensorflow as tf
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


class BASICLSTMCell(RNNCell):
    def __init__(self,
                 num_units,
                 grain,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None):

        super(BASICLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            logging.warn(
                "%s: Using a concatenated state is slower and will soon be "
                "deprecated.  Use state_is_tuple=True.", self)
        self._num_units = num_units
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
        concat = _line_sep([inputs, h], 4 * self._num_units, bias=True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(
            value=concat, num_or_size_splits=4, axis=1)

        new_c = (c * sigmoid(f + self._forget_bias) +
                 sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


def _line_sep(args,
              output_size,
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

        xh = tf.matmul(x, W_xh)

        ih = tf.matmul(h, W_ih) + cn_xh[:, :h_size]
        jh = tf.matmul(h, W_jh) + cn_xh[:, h_size:h_size * 2]
        fh = tf.matmul(h, W_fh) + cn_xh[:, h_size * 2:h_size * 3]
        oh = tf.matmul(h, W_oh) + cn_xh[:, h_size * 3:]

        if not bias:
            return ih, jh, fh, oh
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(
                    0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


def identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = np.identity(size) * scale
        t[:, size * 2:size * 3] = np.identity(size) * scale
        t[:, size * 3:] = np.identity(size) * scale
        return tf.constant(t, dtype)

    return _initializer
