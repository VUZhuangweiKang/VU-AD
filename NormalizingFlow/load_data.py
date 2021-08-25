from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
tf1=tf.compat.v1
import tensorflow_probability as tfp
tfd = tfp.distributions

def load_moon_dataset(n_samples=1000):
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
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
    xlim, ylim = [-2, 2], [-2, 2]
    X_data = StandardScaler().fit_transform(X)
    return X_data.astype(np.float32)


def load_smd_dataset(group, entity):
    SMD_BASE_PATH = 'Dataset/SMD'

    X_train = load_data('%s/train/machine-%d-%d.txt' % (SMD_BASE_PATH, group, entity), header=False)
    X_train.columns = ['m%d' % i for i in range(X_train.shape[1])]
    X_train.index = pd.date_range('2021/03/02', '2021/03/21', periods=X_train.shape[0])
    X_train.index.name = 'timestamp'

    X_test = load_data('%s/test/machine-%d-%d.txt' % (SMD_BASE_PATH, group, entity), header=False)
    X_test.columns = ['m%d' % i for i in range(X_test.shape[1])]
    X_test.index = pd.date_range('2021/03/21', '2021/4/8', periods=X_test.shape[0])
    X_test.index.name = 'timestamp'

    y_true = pd.read_csv('Dataset/SMD/test_label/machine-%d-%d.txt' % (group, entity), header=None)
    y_true.columns = ['label']
    y_true.index = X_test.index