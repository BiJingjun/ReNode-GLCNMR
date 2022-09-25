import numpy as np
import scipy.sparse as sp
import sys
import scipy.io as sio
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str,path="../data/"):
    path = path + "scence-750/"

    features = sio.loadmat(path + "feat")
    features = features['feat']
    adj = sio.loadmat(path + "adj")
    adj = adj['adj']
    labels = sio.loadmat(path + "label")
    labels = labels['label']
    trainmask = sio.loadmat(path + dataset_str + "trid")
    idx_train = trainmask['train_idx']
    valmask = sio.loadmat(path + dataset_str + "vaid")
    idx_val = valmask['valid_idx']
    testmask = sio.loadmat(path + dataset_str + "teid")
    idx_test = testmask['test_idx']

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(abs(features).sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    edge = np.array(np.nonzero(adj_normalized.todense()))
    return sparse_to_tuple(adj_normalized), edge



def preprocess_laplacian(adj):
    """Preprocessing of laplacian"""
    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian



def construct_feed_dict(features, adj, labels, labels_mask, epoch, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['step']: epoch})
    feed_dict.update({placeholders['num_nodes']: features[2][0]})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

