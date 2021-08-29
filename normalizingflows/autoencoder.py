from abc import ABC
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import Input, layers, Model


def build_encoder(input_dim, latent_dim):
    input_layer = layers.Input((input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(64, activation='relu')(encoded)
    output_layer = layers.Dense(latent_dim, activation='relu')(encoded)
    
    encoder = Model(input_layer, output_layer, name='encoder')
    return encoder


def build_decoder(input_dim, latent_dim):
    input_layer = layers.Input((latent_dim,))
    decoded = layers.Dense(64, activation='relu')(input_layer)
    decoded = layers.Dense(128, activation='relu')(decoded)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)
    
    decoder = Model(input_layer, output_layer, name='decoder')
    return decoder


class AutoEncoder(Model):
    def __init__(self,
                 encoder,
                 decoder,
                 learning_rate=1e-3,
                 immediate_operations=None,
                 **kwargs):
        super(AutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.immediate_operations = immediate_operations
        self.decoder = decoder
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.total_loss_tracker = tfk.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tfk.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tfk.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, data):
        latent_space = self.encoder(data)
        reconstructed = self.decoder(latent_space)
        return reconstructed

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            if self.immediate_operations is not None:
                z = self.immediate_operations(z)
            reconstruction = self.decoder(z)
            reconstruction_loss = tfk.losses.mean_squared_error(data, reconstruction)
            kl_loss = tf.losses.kl_divergence(data, reconstruction)
            total_loss = reconstruction_loss + kl_loss
        
        grads = tape.gradient(reconstruction_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }