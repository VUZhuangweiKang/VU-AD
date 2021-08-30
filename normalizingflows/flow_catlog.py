import numpy as np
import tensorflow as tf
tf1=tf.compat.v1
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import regularizers

tfd = tfp.distributions
tfb = tfp.bijectors


def MAF(base_dist, num_bijectors, hidden_units, ndims, activation=tf.nn.relu):
    bijectors = []
    for i in range(num_bijectors):
        made = tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units, activation=activation)
        maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
        permute = tfb.Permute(permutation=np.arange(ndims)[::-1])
        bijectors.append(maf)
        bijectors.append(permute)

    # Discard the last Permute layer.
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    flow = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
    return flow


def IAF(base_dist, num_bijectors, hidden_units, ndims, activation=tf.nn.relu):
    bijectors = []
    for i in range(num_bijectors):
        made = tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units, activation=activation)
        maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
        permute = tfb.Permute(permutation=np.arange(ndims)[::-1])
        bijectors.append(tfb.Invert(maf))
        bijectors.append(permute)

    # Discard the last Permute layer.
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    flow = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
    return flow


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