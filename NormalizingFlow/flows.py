import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
from tensorflow.keras import regularizers

tfd = tfp.distributions
tfb = tfp.bijectors


def MAF(base_dist, num_bijectors, hidden_units, activation=tf.nn.relu):
    bijectors = []
    for i in range(num_bijectors):
        made = tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units, activation=activation)
        maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
        permute = tfb.Permute(permutation=[1, 0])
        bijectors.append(maf)
        bijectors.append(permute)

    # Discard the last Permute layer.
    flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
    flow = tfd.TransformedDistribution(distribution=base_dist, bijector=flow_bijector)
    return flow


def IAF(num_bijectors, hidden_units, activation=tf.nn.relu):
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], tf.float32), scale_diag=tf.ones([2], tf.float32))
    bijectors = []
    for i in range(num_bijectors):
        made = tfb.AutoregressiveNetwork(params=2, hidden_units=hidden_units, activation=activation)
        maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
        permute = tfb.Permute(permutation=[1, 0])
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