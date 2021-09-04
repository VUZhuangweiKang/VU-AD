from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import tensorflow as tf
tf1=tf.compat.v1
import tensorflow_probability as tfp
tfd = tfp.distributions


def load_moon_dataset(n_samples=1000, noise=0.05):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
    X, y = noisy_moons
    X_data = StandardScaler().fit_transform(X)
    return X_data.astype(np.float32)


def load_single_moon_dataset(samples=1000, batch_size=512):
    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(batch_size)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(batch_size, dtype=tf.float32))
    x1_samples = x1.sample()
    X_data = tf.stack([x1_samples, x2_samples], axis=1)
    try:
        X_data = X_data.numpy()
    except:
        sess = tf1.InteractiveSession()
        sess.run(tf1.global_variables_initializer())
        X_data = X_data.eval(session=sess)
    X_data = StandardScaler().fit_transform(X_data)
    return X_data.astype(np.float32)


def load_gaussian(n_samples=2000):
    mean = [0.4, 1]
    A = np.array([[2, .3], [-1., 4]])
    cov = A.T.dot(A)
    X = np.random.multivariate_normal(mean, cov, n_samples)
    X_data = StandardScaler().fit_transform(X)
    return X_data.astype(np.float32)


def load_smd_dataset(group=1, entity=3, select_dims=None):
    SMD_BASE_PATH = '../Dataset/SMD'
    X_data = pd.read_csv('%s/train/machine-%d-%d.txt' % (SMD_BASE_PATH, group, entity), header=None)
    if select_dims is None:
        select_dims = np.random.randint(low=0, high=X_data.shape[1], size=2)
    else:
        assert type(select_dims) is list
    X_data = X_data.iloc[:, select_dims]
    X_data = StandardScaler().fit_transform(X_data)
    y = pd.read_csv('%s/test_label/machine-%d-%d.txt' % (SMD_BASE_PATH, group, entity), header=None)
    return X_data.astype(np.float32), y.astype(np.float32)