import tensorflow as tf
import tensorflow.keras as tfk
tf1=tf.compat.v1
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors


"""
def train_dist_routine_(X_data, trainable_dist, n_epochs=200, batch_size=None, n_disp=100, lr=1e-3):
    x_ = Input(shape=(X_data.shape[1],))
    log_prob_ = trainable_dist.log_prob(x_)
    model = Model(x_, log_prob_)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=lambda _, log_prob: -log_prob)

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns
    
    epoch_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                    print('\n Epoch {}/{}'.format(epoch+1, n_epochs, logs),
                                            '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                    if epoch % n_disp == 0 else False)

    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.2,
                        shuffle=True,
                        verbose=False,
                        callbacks=[epoch_callback])
    return history
"""


@tf.function
def train_step(X_data, optimizer, trainable_dist): 
    with tf.GradientTape() as tape:
        tape.watch(trainable_dist.trainable_variables)
        nll = -tf.reduce_mean(trainable_dist.log_prob(X_data))

        # tf.print(tf.reduce_mean(nll), tf.reduce_min(nll), tf.reduce_max(nll))
        # threshold = 3
        # outliers = tf.cast(tf.gather(nll, tf.where(tf.less(nll, threshold))[:, 0]), tf.float32)
        # if tf.shape(outliers)[0] == 0:
        #     loss = tf.reduce_mean(nll)
        # else:
        #     loss = tf.reduce_mean(nll) - tf.reduce_mean(outliers)

        loss = nll

    gradients = tape.gradient(loss, trainable_dist.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_dist.trainable_variables))
    return loss

def train_dist_routine(X_data, flow, learning_rate=1e-3, steps=1000):
    -tf.reduce_mean(flow.log_prob(X_data)) 
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)
    losses = []
    for i in range(steps):
        loss = train_step(X_data, optimizer, flow)
        losses.append(loss)
        if (i % 100 == 0):
            print('steps:', i, "\t loss:",loss.numpy())
    return losses


def train_dist_routine_v1(X_data, trainable_dist, steps=int(1e4), learning_rate=1e-3):
    global_step = []
    np_losses = []

    sess = tf1.InteractiveSession()
    sess.run(tf1.global_variables_initializer())
    loss = -tf1.reduce_mean(trainable_dist.log_prob(X_data))
    train_op = tf1.train.AdamOptimizer(learning_rate).minimize(loss)
    sess.run(tf1.global_variables_initializer())

    for i in range(steps):
        _, np_loss = sess.run([train_op, loss])
        if i % 500 == 0:
            global_step.append(i)
            np_losses.append(np_loss)
        if i % int(1000) == 0:
            print(i, np_loss)
    return np_losses