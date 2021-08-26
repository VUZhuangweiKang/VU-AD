import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import tensorflow as tf
import tensorflow.keras as tfk
tf1=tf.compat.v1
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score


# Define a plot contour routine

def plot_contour_prob(dist, rows=1, title=[''], scale_fig=4):
    cols = int(len(dist) / rows)
    xx = np.linspace(-5.0, 5.0, 100)
    yy = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(xx, yy)

    fig, ax = plt.subplots(rows, cols, figsize=(scale_fig * cols, scale_fig * rows))
    fig.tight_layout(pad=4.5)

    i = 0
    for r in range(rows):
        for c in range(cols):
            Z = dist[i].prob(np.dstack((X, Y)))
            if len(dist) == 1:
                axi = ax
            elif rows == 1:
                axi = ax[c]
            else:
                axi = ax[r, c]

            # Plot contour
            p = axi.contourf(X, Y, Z)

            # Add a colorbar
            divider = make_axes_locatable(axi)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(p, cax=cax)

            # Set title and labels
            axi.set_title('Filled Contours Plot: ' + str(title[i]))
            axi.set_xlabel('x')
            axi.set_ylabel('y')

            i += 1
    plt.show()



# Define a scatter plot routine for the bijectors
def plot_samples(samples, names, rows=1, legend=False):
    cols = int(len(samples) / rows)
    f, arr = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            res = samples[i]
            try:
                X, Y = res[..., 0].numpy(), res[..., 1].numpy()
            except:
                sess = tf1.InteractiveSession()
                sess.run(tf1.global_variables_initializer())
                X, Y = res[..., 0].eval(session=sess), res[..., 1].eval(session=sess)
            if rows == 1:
                p = arr[c]
            else:
                p = arr[r, c]
            p.scatter(X, Y, s=10, color='red')
            p.set_title(names[i])
            
            i += 1
    plt.show()


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


@tf.function
def train_step(X_data, optimizer, trainable_dist): 
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(trainable_dist.log_prob(X_data)) 
    gradients = tape.gradient(loss, trainable_dist.trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_dist.trainable_variables))
    return loss

def train_dist_routine(X_data, flow, learning_rate=1e-3, steps=1000):
    -tf.reduce_mean(flow.log_prob(X_data)) 
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
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

def plot_loss(train_hist):
    train_losses = train_hist.history['loss']
    valid_losses = train_hist.history['val_loss']
    # Plot loss vs epoch
    plt.plot(train_losses, label='train')
    plt.plot(valid_losses, label='valid')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Negative log likelihood")
    plt.title("Training and validation loss curves")
    plt.show()


# Define a plot routine

def visualize_training_data(X_data, samples):
    f, arr = plt.subplots(1, 2, figsize=(15, 6))
    names = ['Data', 'Trainable']
    samples = [tf.constant(X_data), samples[-1]]

    for i in range(2):
        res = samples[i]
        X, Y = res[..., 0].numpy(), res[..., 1].numpy()
        arr[i].scatter(X, Y, s=10, color='red')
        arr[i].set_xlim([-3, 3])
        arr[i].set_ylim([-3, 3])
        arr[i].set_title(names[i])


# Make samples
def make_samples(base_distribution, trainable_distribution, n_samples=1000, dims=2):
    x = base_distribution.sample((n_samples, dims))
    samples = [x]
    names = [base_distribution.name]
    for bijector in reversed(trainable_distribution.bijector.bijectors):
        x = bijector.forward(x)
        samples.append(x)
        names.append(bijector.name)
    return names, samples


def detect_anomalies(data, model, threshold=None):
    pdf = model.prob(data).numpy()
    if threshold is None:
        z_scores = stats.zscore(pdf)
        anomalies = np.where((z_scores > 2) | (z_scores < -2))
    else:
        anomalies = np.where(pdf < threshold)
    return anomalies

def label_anomalies(X_data, anomaly_index):
    plt.scatter(X_data[:, 0], X_data[:, 1], s=10, color='red', label='X_data')
    plt.scatter(X_data[anomaly_index, 0], X_data[anomaly_index, 1], s=10, color='blue', label='Anomalies')
    plt.legend()
    plt.show()


# sample from the tail(-inf, mu-2sigma) & (mu+2sigma, inf) of a well-known(Gaussian) distribution
def sample_anomalies(flow, n_samples=1000, mu=0, sigma=1, factor=2, sample_shape=2):
    clip_a_l = -np.inf
    clip_b_l = mu - factor * sigma
    clip_a_r = mu + factor * sigma
    clip_b_r = np.inf

    X1_samples = truncnorm.rvs(a=clip_a_l, b=clip_b_l, loc=mu, scale=sigma, size=(n_samples//2, sample_shape))
    X2_samples = truncnorm.rvs(a=clip_a_r, b=clip_b_r, loc=mu, scale=sigma, size=(n_samples//2, sample_shape))
    samples = np.vstack([X1_samples, X2_samples]).astype(np.float32)
    generated_anomalies = flow.bijector.forward(samples).numpy()
    return generated_anomalies

def sample_normals(flow, n_samples=3000, mu=0, sigma=1, factor=2, sample_shape=2):
    clip_a = mu - factor * sigma
    clip_b = mu + factor * sigma

    a, b = (clip_a - mu) / sigma, (clip_b - mu) / sigma
    samples = truncnorm.rvs(a=a, b=b, loc=mu, scale=sigma, size=(n_samples, sample_shape)).astype(np.float32)
    samples = flow.bijector.forward(samples).numpy()
    return samples


def RFClassifier(X_train, y_train, X_test=None, y_test=None):
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(X_train, y_train)

    score = None
    if X_test and y_test:
        predicted = clf.predict(X_test)
        score = score(y_test, predicted)
    return clf, score
