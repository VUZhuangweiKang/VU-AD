import numpy as np
from scipy.stats import truncnorm


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