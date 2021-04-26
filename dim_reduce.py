from sklearn.decomposition import PCA


def pca_dr(data, n_components, transform=False):
    """
    :param data: X
    :param n_components: dimension of latent factors if n_components is integer, or the amount of variance that needs to be
        explained if n_components is floating
    :param transform: fit and transform input data to new data in latent space
    :return: number of components, explained variance ratio of each factor, explained variance of each factor, cumulative explained variance ratio
    """
    pca = PCA(n_components=n_components)
    if not transform:
        pca.fit(data)
        return pca.n_components_, pca.explained_variance_ratio_, pca.explained_variance_, pca.explained_variance_ratio_
    else:
        new_X = pca.fit_transform(X=data)
        print('Number of factors:', new_X.shape[1])
        print('Explained variance ratio of each factor:', pca.explained_variance_ratio_)
        print('Explained variance of each factor::', pca.explained_variance_)
        print('Total explained variance ratio:', sum(pca.explained_variance_ratio_))
        return new_X