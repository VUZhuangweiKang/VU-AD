import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
tf1=tf.compat.v1
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Dense
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


class AutoRegressiveFlowModel(Model):
    def __init__(self, 
                 base_dist, 
                 num_bijectors, 
                 hidden_units, 
                 ndims, 
                 activation=tf.nn.relu, 
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 use_batchnorm=False,
                 **kwargs):
        super(AutoRegressiveFlowModel, self).__init__(**kwargs)
        self.base_dist = base_dist
        self.num_bijectors = num_bijectors
        self.hidden_units = hidden_units
        self.ndims = ndims
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate, decay=weight_decay)
        self.loss_tracker = tfk.metrics.Mean(name="loss")

    @abstractmethod
    def build_flow(self):
        raise NotImplementedError("Flow builder is not implemented")

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)


class IAF(AutoRegressiveFlowModel):
    def __init__(self, *args, **kwargs):
        super(IAF, self).__init__(*args, **kwargs)
        self.build_flow()

    def build_flow(self):
        bijectors = []
        for i in range(self.num_bijectors):
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.hidden_units, activation=self.activation)
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
            bijectors.append(tfb.Invert(maf))
            if self.use_batchnorm and i % 2 == 0:
                bijectors.append(BatchNorm(name='batch_norm%d' % i))

            permute = tfb.Permute(permutation=np.arange(self.ndims)[::-1])
            bijectors.append(permute)

        # Discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        self.flow = tfd.TransformedDistribution(distribution=self.base_dist, bijector=flow_bijector)
    
    def train_step(self, data): 
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(data)) 
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


class MAF(AutoRegressiveFlowModel):
    def __init__(self, *args, **kwargs):
        super(MAF, self).__init__(*args, **kwargs)
        self.build_flow()

    def build_flow(self):
        bijectors = []
        for i in range(self.num_bijectors):
            made = tfb.AutoregressiveNetwork(params=2, hidden_units=self.hidden_units, activation=self.activation)
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
            bijectors.append(maf)
            if self.use_batchnorm and i % 2 == 0:
                bijectors.append(BatchNorm(name='batch_norm%d' % i))

            permute = tfb.Permute(permutation=np.arange(self.ndims)[::-1])
            bijectors.append(permute)

        # Discard the last Permute layer.
        flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        self.flow = tfd.TransformedDistribution(distribution=self.base_dist, bijector=flow_bijector)

    def train_step(self, data): 
        with tf.GradientTape() as tape:
            nll = self.flow.log_prob(data)

            # tf.print(tf.reduce_mean(nll), tf.reduce_min(nll), tf.reduce_max(nll))
            threshold = -5
            # outliers = tf.cast(tf.gather(nll, tf.where(tf.less(nll, threshold))[:, 0]), tf.float32)
            # loss = tf.reduce_mean(nll) - tf.reduce_mean(outliers)

            loss = -tf.reduce_mean(nll)
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# --------------------------------------------------

# Creating a custom layer with keras API.
def Coupling(input_dims, scale_net_layers=4, shift_net_layers=4, output_dim=256):
    input = Input(shape=input_dims)
    reg = 0.01

    input_ = input
    for i in range(scale_net_layers):
        layer = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input_)
        input_ = layer
    shift_net_out = Dense(input_dims, activation="linear", kernel_regularizer=regularizers.l2(reg))(input_)

    for i in range(shift_net_layers):
        layer = Dense(output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg))(input_)
        input_ = layer
    scale_net_out = Dense(input_dims, activation="tanh", kernel_regularizer=regularizers.l2(reg))(input_)

    return Model(inputs=input, outputs=[scale_net_out, shift_net_out])


class RealNVP(Model):
    def __init__(self, input_shape, num_coupling_layers):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers

        # Distribution of the latent space.
        self.distribution = tfd.MultivariateNormalDiag(loc=tf.zeros([input_shape], tf.float32), scale_diag=tf.ones([input_shape], tf.float32))
        self.masks = np.array(
            [[0, 1], [1, 0]] * (num_coupling_layers // 2), dtype="float32"
        )
        self.loss_tracker = tfk.metrics.Mean(name="loss")
        self.layers_list = [Coupling(input_shape) for i in range(num_coupling_layers)]

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self, x, training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.num_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (
                reversed_mask
                * (x * tf.exp(direction * s) + direction * t * tf.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * tf.reduce_sum(s, [1])

        return x, log_det_inv

    # Log likelihood of the normal distribution plus the log determinant of the jacobian.

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)

    def train_step(self, data):
        with tf.GradientTape() as tape:

            loss = self.log_loss(data)

        g = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(g, self.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self.log_loss(data)
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}


# -----------------------------------------------------

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