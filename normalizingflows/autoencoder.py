from abc import ABC
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import Input, layers, Model
import flow_manager as fm


def build_encoder(input_dim, latent_dim, hidden_units=[128, 64]):
    input_layer = layers.Input((input_dim,))
    encoded = layers.Dense(hidden_units[0], activation='relu')(input_layer)
    for hu in hidden_units[1:]:
        encoded = layers.Dense(hu, activation='relu')(encoded)
    output_layer = layers.Dense(latent_dim, activation='relu')(encoded)
    
    encoder = Model(input_layer, output_layer, name='encoder')
    return encoder


def build_decoder(input_dim, latent_dim, hidden_units=[64, 128]):
    input_layer = layers.Input((latent_dim,))
    decoded = layers.Dense(hidden_units[0], activation='relu')(input_layer)
    for hu in hidden_units[1:]:
        decoded = layers.Dense(hu, activation='relu')(decoded)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    decoder = Model(input_layer, output_layer, name='decoder')
    return decoder


class AutoEncoder(Model):
    def __init__(self,
                 encoder,
                 decoder,
                 learning_rate=1e-3,
                 flow_model=None,
                 flow_opt=None,
                 **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.reconstruction_loss_tracker = tfk.metrics.Mean(name="reconstruction_loss")

        self.flow_model = flow_model
        self.flow_opt = flow_opt

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker]

    def call(self, data):
        latent_space = self.encoder(data)
        reconstructed = self.decoder(latent_space)
        return reconstructed

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            if self.flow_model is not None:
                nf_loss = fm.train_step(z, self.flow_opt, self.flow_model)
                z = self.flow_model.bijector.forward(z)
            reconstruction = self.decoder(z)
            reconstruction_loss = tfk.losses.mean_squared_error(data, reconstruction)

        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {"reconstruction_loss": self.reconstruction_loss_tracker.result()}