import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
sns.set(style="whitegrid")
tfd = tf.contrib.distributions
tfb = tfd.bijectors

batch_size=512
DTYPE=tf.float32
NP_DTYPE=np.float32


def load_dummy_dataset():
    DATASET = 1
    if DATASET == 0:
        mean = [0.4, 1]
        A = np.array([[2, .3], [-1., 4]])
        cov = A.T.dot(A)
        X = np.random.multivariate_normal(mean, cov, 2000)
        plt.scatter(X[:, 0], X[:, 1], s=10, color='red')
        dataset = tf.data.Dataset.from_tensor_slices(X.astype(NP_DTYPE))
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=X.shape[0])
        dataset = dataset.prefetch(3 * batch_size)
        dataset = dataset.batch(batch_size)
        data_iterator = dataset.make_one_shot_iterator()
        X_train = data_iterator.get_next()
    elif DATASET == 1:
        x2_dist = tfd.Normal(loc=0., scale=4.)
        x2_samples = x2_dist.sample(batch_size)
        x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                        scale=tf.ones(batch_size, dtype=DTYPE))
        x1_samples = x1.sample()
        x_samples = tf.stack([x1_samples, x2_samples], axis=1)
        X_train = sess.run(x_samples)
        plt.scatter(X_train[:, 0], X_train[:, 1], s=10)
        plt.xlim([-5, 30])
        plt.ylim([-10, 10])