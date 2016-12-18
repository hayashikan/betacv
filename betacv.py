from sklearn.metrics import pairwise_distances
from sklearn.metrics import euclidean_distances
from math import ceil
import numpy as np
import time


def my_betacv_simple(data, labels, size=3000, metric='euclidean'):
    n = labels.shape[0]
    n_slices = ceil(n/size)
    intra = 0
    inter = 0
    n_in = 0
    n_out = 0
    last = 0
    labels_unq = np.unique(labels)
    members = np.array([member_count(labels, i) for i in labels_unq])
    N_in = np.array([i*(i-1) for i in members])
    n_in = np.sum(N_in)
    N_out = np.array([i*(n-i) for i in members])
    n_out = np.sum(N_out)
    
    for i in range(n_slices):
        x = data[last:(last+size), :]
        distances = euclidean_distances(x, data)
        j_range = min(size, n-size*i)
        A = np.array([intra_cluster_distance(distances[j], labels, j+last)
                  for j in range(j_range)])
        B = np.array([inter_cluster_distance(distances[j], labels, j+last)
                  for j in range(j_range)])
        intra += np.sum(A)
        inter += np.sum(B)
        last += size

    betacv = (intra/n_in)/(inter/n_out)
    print('simple intra:', intra)
    print('simple inter:', inter)
    print('simple n_in :', n_in)
    print('simple n_out:', n_out)
    return betacv

def my_betacv(data, labels, metric='euclidean'):
    distances = pairwise_distances(data, metric=metric)
    n = labels.shape[0]
    A = np.array([intra_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    B = np.array([inter_cluster_distance(distances[i], labels, i)
                  for i in range(n)])
    a = np.sum(A)
    b = np.sum(B)
    labels_unq = np.unique(labels)
    members = np.array([member_count(labels, i) for i in labels_unq])
    N_in = np.array([i*(i-1) for i in members])
    n_in = np.sum(N_in)
    N_out = np.array([i*(n-i) for i in members])
    n_out = np.sum(N_out)
    betacv = (a/n_in)/(b/n_out)
    # print('intra:', a)
    # print('inter:', b)
    # print('n_in :', n_in)
    # print('n_out:', n_out)
    return betacv

def intra_cluster_distance(distances_row, labels, i):
    mask = labels == labels[i]
    mask[i] = False
    if not np.any(mask):
        # cluster of size 1
        return 0
    a = np.sum(distances_row[mask])
    return a

def inter_cluster_distance(distances_row, labels, i):
    mask = labels != labels[i]
    b = np.sum(distances_row[mask])
    return b

def member_count(labels, i):
    mask = labels == i
    return len(labels[mask])

