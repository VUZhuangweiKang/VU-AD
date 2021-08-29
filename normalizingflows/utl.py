import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import truncnorm
import tensorflow as tf
tf1=tf.compat.v1
import numpy as np


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


# plot temporal data, the index(timestamp) shoud be provided
def pyplot_X_data_ts(X, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.index, y=X[col], mode='lines', name=col))
    fig.update_layout(height=400, width=800, showlegend=True)
    fig.show()


def pyplot_anomaly_ts(X, y, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.index, y=X[col], mode='lines', name='y_test'))
    fig.add_trace(go.Scatter(x=y[y['label'] == 1].index, y=X[y['label'] == 1][col], mode='markers', name='Anomaly'))
    fig.update_layout(showlegend=True, xaxis_title="Time", yaxis_title="value", height=400, width=800)
    fig.show()