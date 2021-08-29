import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.decomposition import PCA


def process_smd(g, e):
    X_train = pd.read_csv('Dataset/SMD/train/machine-%d-%d.txt' % (g, e), header=None)
    X_train.columns = ['m%d' % i for i in range(X_train.shape[1])]
    # create dummy timestamp as index
    X_train.index = pd.date_range('2021/03/02', '2021/03/21', periods=X_train.shape[0])
    X_train.index.name = 'timestamp'
    X_train.to_csv('Dataset/SMD/processed/train/machine-%d-%d.txt' % (g, e))

    X_test = pd.read_csv('Dataset/SMD/test/machine-%d-%d.txt' % (g, e), header=None)
    X_test.columns = ['m%d' % i for i in range(X_test.shape[1])]
    X_test.index = pd.date_range('2021/03/21', '2021/4/8', periods=X_test.shape[0])
    X_test.index.name = 'timestamp'
    X_test.to_csv('Dataset/SMD/processed/test/machine-%d-%d.txt' % (g, e))

    return X_train, X_test


def load_data(path, header=False):
    if not header:
        data = pd.read_csv(path, index_col=0, header=None)
    else:
        data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data


def minmax_scaler(X_train, X_test=None):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['m%d' % i for i in range(X_train.shape[1])])
    if X_test:
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=['m%d' % i for i in range(X_test.shape[1])])
    return X_train, X_test

# remove all-zero columns that are in training or testing set
def rm_zero_cols(X_train, X_test=None):
    train_nonzero_cols = X_train.columns[(X_train != 0).any()]
    if X_test:
        test_nonzero_cols = X_test.columns[(X_test != 0).any()]
        cols = set(train_nonzero_cols) & set(test_nonzero_cols)
        return X_train[cols], X_test[cols]
    return X_train[train_nonzero_cols]


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

def slide_window_detector(detector, X_train, X_test, sw, fs, train_contextual=False):
    test_start = 0
    train_results = []
    test_results = []
    while test_start+fs < X_test.shape[0]:
        if train_contextual:
            sw_train = pd.DataFrame([])
            i = 0
            while i < X_train.shape[0]:
                sw_train = sw_train.append(X_train[i:i+fs])
                i += sw
        else:
            sw_train = X_train

        sw_test = X_test.iloc[test_start : test_start+fs]

        train_scores, test_scores = detector(sw_train, sw_test)
        train_results.append(train_scores)
        test_results.append(test_scores)
        test_start += fs

        # move training dataset
        X_train = X_train[test_start:]
        X_train = X_train.append(X_test[:test_start])
    return train_results, test_results


# def create_sequence(values, time_steps):
#     output = []
#     for i in range(len(values) - time_steps + 1):
#         output.append(values[i : (i + time_steps)])
#     return np.stack(output)


# def slide_window_detector(detector, X_train, X_test, time_steps):
#     x_train = create_sequence(X_train, time_steps)
#     x_test = create_sequence(X_test, time_steps)
#     train_scores, test_scores = detector(x_train, x_test)
#     return train_scores, test_scores


def label_anomalies_1(y_train, y_test, sw, fs, alph, eval_opt='global'):
    results = pd.DataFrame([])
    for i in range(len(y_train)):
        temp = pd.DataFrame([])
        temp['anomaly_score'] = y_test[i]
        g_mu = np.mean(y_train[i])
        g_sigma = np.std(y_train[i])
        context_scores = []
        j = 0
        while j+fs < len(y_train[i]):
            context_scores.extend(y_train[i][j:j+fs])
            j += sw
        c_mu = np.mean(context_scores)
        c_sigma = np.std(context_scores)
        if eval_opt == 'hybrid':
            temp['label'] = temp['anomaly_score'] > min(c_mu+alph*c_sigma, g_mu+alph*g_sigma)
        elif eval_opt == 'global':
            temp['label'] = temp['anomaly_score'] > g_mu+alph*g_sigma
        elif eval_opt == 'contextual':
            temp['label'] = temp['anomaly_score'] > c_mu+alph*c_sigma
        results = results.append(temp)
    return results


def label_anomalies_2(y_test, y_test_label, sw, fs, alph, eval_opt='hybrid'):
    results = pd.DataFrame([])
    for i in range(len(y_test)):
        gscores = []
        for j in range(0, i-1):
            temp = np.array(y_test_label[j])
            gscores.extend(y_test[j][np.where(temp == 0)])
        
        if len(gscores) == 0:
            g_mu = 10000
            g_sigma = 0
        else:
            g_mu = np.mean(gscores)
            g_sigma = np.std(gscores)
        alph = 2
        g_threshold = g_mu+alph*g_sigma

        cscores = []
        j = i - int(sw/fs)
        while j >= 0:
            temp = np.array(y_test_label[j])
            cscores.extend(y_test[j][np.where(temp==0)])
            j -= int(sw/fs)
        if len(cscores) == 0:
            c_mu = 10000
            c_sigma = 0
        else:
            c_mu = np.mean(cscores)
            c_sigma = np.std(cscores)
        c_threshold = c_mu+alph*c_sigma
        
        temp = pd.DataFrame([])
        temp['anomaly_score'] = y_test[i]
        if eval_opt == 'hybrid':
            temp['label'] = temp['anomaly_score'] > min(g_threshold, c_threshold)
        elif eval_opt == 'global':
            temp['label'] = temp['anomaly_score'] > g_threshold
        elif eval_opt == 'contextual':
            temp['label'] = temp['anomaly_score'] > c_threshold
        results = results.append(temp)
    return results