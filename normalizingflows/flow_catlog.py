import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
tf1=tf.compat.v1
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from abc import ABCMeta, abstractmethod

tfd = tfp.distributions
tfb = tfp.bijectors


class BatchNorm(tfb.Bijector):
    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        n = x.get_shape().as_list()[1]
        self.beta = tf.Variable(initial_value=tf.zeros(shape=[1, n], dtype=tf.float32), name='beta', trainable=True)
        self.gamma = tf.Variable(initial_value=tf.zeros(shape=[1, n], dtype=tf.float32), name='gamma', trainable=True)
        self.train_m = tf.Variable(initial_value=tf.zeros(shape=[1, n], dtype=tf.float32), name='mean', trainable=False)
        self.train_v = tf.Variable(initial_value=tf.ones(shape=[1, n], dtype=tf.float32), name='var', trainable=False)
        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq 22. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)
        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keepdims=True)
        # update train statistics via exponential moving average
        update_train_m = self.train_m.assign_sub(self.decay * (self.train_m - m))
        update_train_v = self.train_v.assign_sub(self.decay * (self.train_v - v))
        # normalize using current minibatch statistics, followed by BN scale and shift
        with tf.control_dependencies([update_train_m, update_train_v]):
            return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
        _, v = tf.nn.moments(x, axes=[0], keepdims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.math.log(v + self.eps))
        return abs_log_det_J_inv


class FlowModel(Model):
    def __init__(self, 
                 base_dist, 
                 num_bijectors, 
                 hidden_units, 
                 ndims, 
                 activation=tf.nn.relu, 
                 learning_rate=1e-3,
                 use_batchnorm=False,
                 **kwargs):
        super(FlowModel, self).__init__(**kwargs)
        self.base_dist = base_dist
        self.num_bijectors = num_bijectors
        self.hidden_units = hidden_units
        self.ndims = ndims
        self.activation = activation
        self.use_batchnorm = use_batchnorm

        max_epochs = int(15e3)
        learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(learning_rate, max_epochs, learning_rate/10, power=0.5)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn)
        self.loss_tracker = tfk.metrics.Mean(name="loss")

        self.flow = None

    @abstractmethod
    def build_flow(self):
        raise NotImplementedError("Flow builder is not implemented")

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)
    
    @tf.function
    def train_step(self, data): 
        with tf.GradientTape() as tape:
            tape.watch(self.flow.trainable_variables)

            threshold = -3.
            nll = -self.flow.log_prob(data)
            outliers = tf.cast(tf.gather(nll, tf.where(tf.less(nll, threshold))[:, 0]), tf.float32)
            if tf.shape(outliers)[0] == 0:
                loss = tf.reduce_mean(nll)
            else:
                loss = tf.reduce_mean(nll) - tf.reduce_mean(outliers)

            # loss = -tf.reduce_mean(self.flow.log_prob(data))
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


'''--------------------------------------------- Inverse Autoregressive Flow -----------------------------------------------'''

class IAF(FlowModel):
    def __init__(self, *args, **kwargs):
        super(IAF, self).__init__(*args, **kwargs)
        self.build_flow()

    def build_flow(self):
        bijectors = []
        permutation = tf.cast(np.concatenate((np.arange(self.ndims / 2, self.ndims), np.arange(0, self.ndims / 2))), tf.int32)
        for i in range(self.num_bijectors):
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.hidden_units, activation=self.activation)
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
            bijectors.append(tfb.Invert(maf))
            if self.use_batchnorm and i % 2 == 0:
                bijectors.append(BatchNorm(name='batch_norm%d' % i))
            bijectors.append(tfb.Permute(permutation=permutation))

        # Discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(bijectors)))
        self.flow = tfd.TransformedDistribution(distribution=self.base_dist, bijector=flow_bijector)


'''--------------------------------------------- Masked Autoregressive Flow -----------------------------------------------'''

class Made(tfk.layers.Layer):
    def __init__(self, params, hidden_units=None, activation=None, use_bias=True,
                 kernel_regularizer=None, bias_regularizer=None, name="made"):

        super(Made, self).__init__(name=name)

        self.params = params
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.network = tfb.AutoregressiveNetwork(params=params, hidden_units=hidden_units, activation=activation, 
                                                 use_bias=use_bias, kernel_regularizer=kernel_regularizer, 
                                                 bias_regularizer=bias_regularizer)

    def call(self, x):
        shift, log_scale = tf.unstack(self.network(x), num=2, axis=-1)

        return shift, tf.math.tanh(log_scale)


class MAF(FlowModel):
    def __init__(self, *args, **kwargs):
        super(MAF, self).__init__(*args, **kwargs)
        self.build_flow()

    def build_flow(self):
        bijectors = []
        bijectors.append(BatchNorm(eps=10e-5, decay=0.95))
        permutation = tf.cast(np.concatenate((np.arange(self.ndims / 2, self.ndims), np.arange(0, self.ndims / 2))), tf.int32)
        for i in range(self.num_bijectors):
            made = Made(params=2, hidden_units=self.hidden_units, activation=self.activation)
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made, validate_args=True)
            bijectors.append(maf)
            bijectors.append(tfb.Permute(permutation=permutation))
            if self.use_batchnorm and (i+1) % 2 == 0:
                bijectors.append(BatchNorm(eps=10e-5, decay=0.95))

        flow_bijector = tfb.Chain(list(reversed(bijectors)))
        self.flow = tfd.TransformedDistribution(distribution=self.base_dist, bijector=flow_bijector)


'''--------------------------------------------- Real NVP -----------------------------------------------'''

class NN(Layer):
    """
    Neural Network Architecture for calcualting s and t for Real-NVP
    
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    :param activation: Activation of the hidden units
    """
    def __init__(self, input_shape, n_hidden=[512, 512], activation="relu", name="nn"):
        super(NN, self).__init__(name="nn")
        layer_list = []
        for i, hidden in enumerate(n_hidden):
            layer_list.append(Dense(hidden, activation=activation))
        self.layer_list = layer_list
        self.log_s_layer = Dense(input_shape, activation="tanh", name='log_s')
        self.t_layer = Dense(input_shape, name='t')

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t


class RealNVPBijector(tfb.Bijector):
    """
    Implementation of a Real-NVP for Denisty Estimation. L. Dinh “Density estimation using Real NVP,” 2016.
    This implementation only works for 1D arrays.
    :param input_shape: shape of the data coming in the layer
    :param hidden_units: Python list-like of non-negative integers, specifying the number of units in each hidden layer.
    """

    def __init__(self, input_shape, n_hidden=[512, 512], forward_min_event_ndims=1, validate_args: bool = False, name="real_nvp"):
        super(RealNVPBijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )

        assert input_shape % 2 == 0
        input_shape = input_shape // 2
        nn_layer = NN(input_shape, n_hidden)
        x = tf.keras.Input(input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name="nn")
        
    def _bijector_fn(self, x):
        log_s, t = self.nn(x)
        return tfb.affine_scalar.AffineScalar(shift=t, log_scale=log_s)

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        y_a = self._bijector_fn(x_b).forward(x_a)
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        x_a = self._bijector_fn(y_b).inverse(y_a)
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        return self._bijector_fn(x_b).forward_log_det_jacobian(x_a, event_ndims=1)
    
    def _inverse_log_det_jacobian(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        return self._bijector_fn(y_b).inverse_log_det_jacobian(y_a, event_ndims=1)


class RealNVP(FlowModel):
    def __init__(self, *args, **kwargs):
        super(RealNVP, self).__init__(*args, **kwargs)
        self.build_flow()

    def build_flow(self):
        permutation = tf.cast(np.concatenate((np.arange(self.ndims / 2, self.ndims), np.arange(0, self.ndims / 2))), tf.int32)
        bijectors = []
        for i in range(self.num_bijectors):
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(RealNVPBijector(input_shape=self.ndims, n_hidden=self.hidden_units, name='real_nvp%d' % i))
            bijectors.append(tfp.bijectors.Permute(permutation, name='permutation%d' % i))

        flow_bijector = tfb.Chain(list(reversed(bijectors)))
        self.flow = tfd.TransformedDistribution(
            distribution=self.base_dist, 
            bijector=flow_bijector)


'''--------------------------------------------- PReLU -----------------------------------------------'''

class PReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="prelu"):
        super(PReLU, self).__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf1.where(tf1.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf1.where(tf1.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        I = tf1.ones_like(y)
        J_inv = tf1.where(tf1.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        log_abs_det_J_inv = tf1.log(tf1.abs(J_inv))
        return log_abs_det_J_inv


"""
Y = g(X; shift, scale) = scale @ X + shift
scale = (
  scale_identity_multiplier * tf.diag(tf.ones(d)) +
  tf.diag(scale_diag) +
  scale_tril +
  scale_perturb_factor @ diag(scale_perturb_diag) @
    tf.transpose([scale_perturb_factor])
)
"""
def mlp_flow(base_dist, input_dims, num_bijectors):
    bijectors = []
    
    k = input_dims
    for i in range(num_bijectors):
        with tf1.variable_scope('bijector_%d' % i):
            d = k
            V = tf1.get_variable('V%d'%i, [k, k], dtype=tf1.float32)  # factor loading
            shift = tf1.get_variable('shift%d'%i, [k], dtype=tf1.float32)  # affine shift
            L = tf1.get_variable('L%d'%i, [int(k * (k + 1) / 2)], dtype=tf1.float32)  # lower triangular
            bijectors.append(tfb.Affine(
                scale_tril=tfp.math.fill_triangular(L),
                scale_perturb_factor=V,
                shift=shift,
            ))
            alpha = tf1.abs(tf1.get_variable('alpha%d'%i, [], dtype=tf1.float32)) + .01
            bijectors.append(PReLU(alpha=alpha, name='prelu%d' % i))

    mlp_bijector = tfb.Chain(list(reversed(bijectors[:-1])), name='mlp_bijector')
    flow = tfd.TransformedDistribution(
        distribution=base_dist,
        bijector=mlp_bijector
    )
    return flow