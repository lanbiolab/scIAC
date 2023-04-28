import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, find
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA,NMF,FastICA,FactorAnalysis,LatentDirichletAllocation,TruncatedSVD
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
import time
import argparse

from sklearn.feature_extraction.text import TfidfTransformer

def stimpute(data, n_components=20, random_pca=True,
          t=6, k=30, ka=10, epsilon=1, rescale=99):
    # data = tf_idf_transform(data)
    # data = np.array(data.todense)
    # transformer = TfidfTransformer()
    # data = transformer.fit_transform(data).toarray()
    if n_components != None:
        print('doing PCA')
        projected_data = run_pca(data, n_components=n_components, random=random_pca)
        # model = NMF(n_components=n_components,
        #         random_state=42,
        #         init='nndsvd',
        #         alpha=1.0,
        #         l1_ratio=0,
        #         max_iter=500,
        #         verbose=0)
        #
        # projected_data = model.fit_transform(X=data)

    else:
        projected_data = data

    # run diffusion maps to get markov matrix
    L = compute_markov(projected_data, k=k, epsilon=epsilon,
                       distance_metric='euclidean', ka=ka)

    # remove tsne kernel for now
    # else:
    #     distances = pairwise_distances(pca_projected_data, squared=True)
    #     if k_knn > 0:
    #         neighbors_nn = np.argsort(distances, axis=0)[:, :k_knn]
    #         P = _joint_probabilities_nn(distances, neighbors_nn, perplexity, 1)
    #     else:
    #         P = _joint_probabilities(distances, perplexity, 1)
    #     P = squareform(P)

    #     #markov normalize P
    #     L = np.divide(P, np.sum(P, axis=1))

    # get imputed data matrix -- by default use data_norm but give user option to pick
    new_data, L_t = impute_fast(data, L, t, rescale_percent=rescale)

    return new_data


def run_pca(data, n_components=8, random=True):
    solver = 'randomized'
    if random != True:
        solver = 'full'

    # pca = PCA(n_components=n_components, svd_solver=solver)
    pca = FastICA(n_components=n_components,random_state=0,whiten=True)
    # pca = FactorAnalysis(n_components=n_components)
    # pca = TruncatedSVD(n_components=n_components,random_state=0)
    # pca = TSNE(n_components=2,random_state=0)
    return pca.fit_transform(data)

def tf_idf_transform(data):
    print('TF-IDF processing')
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(data))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(data)))

    return tf_idf


def impute_fast(data, L, t, rescale_percent=0, L_t=None, tprev=None):
    # convert L to full matrix
    if issparse(L):
        L = L.todense()

    # L^t
    print('MAGIC: L_t = L^t')
    if L_t == None:
        L_t = np.linalg.matrix_power(L, t)
    else:
        L_t = np.dot(L_t, np.linalg.matrix_power(L, t - tprev))

    print('MAGIC: data_new = L_t * data')
    data_new = np.array(np.dot(L_t, data))

    # rescale data
    if rescale_percent != 0:
        if len(np.where(data_new < 0)[0]) > 0:
            print('Rescaling should not be performed on log-transformed '
                  '(or other negative) values. Imputed data returned unscaled.')
            return data_new, L_t

        M99 = np.percentile(data, rescale_percent, axis=0)
        M100 = data.max(axis=0)
        indices = np.where(M99 == 0)[0]
        M99[indices] = M100[indices]
        M99_new = np.percentile(data_new, rescale_percent, axis=0)
        M100_new = data_new.max(axis=0)
        indices = np.where(M99_new == 0)[0]
        M99_new[indices] = M100_new[indices]
        max_ratio = np.divide(M99, M99_new)
        data_new = np.multiply(data_new, np.tile(max_ratio, (len(data), 1)))

    return data_new, L_t


def compute_markov(data, k=10, epsilon=1, distance_metric='euclidean', ka=0,alpha=1):
    N = data.shape[0]  # 细胞数
    M = data.shape[1]
    alpha = 1
    # Nearest neighbors
    print('Computing distances')
    nbrs = NearestNeighbors(n_neighbors=k, metric=distance_metric).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # weight = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(i+1,N):
    #         nni = indices[i,:]
    #         nnj = indices[j, :]
    #         shared = np.intersect1d(nni,nnj)
    #         s= [0]
    #         s = np.array(s)
    #         for l in range(len(shared)):
    #             s = np.append(s,k-0.5*((np.where(nni==shared[l])[0])+(np.where(nnj==shared[l])[0])))
    #         weight[i,j] = np.max(s)
    #         weight[j,i] = weight[i,j]
    # W = weight





    ##自适应高斯核，distance先除sigma
    if ka > 0:
        print('Autotuning distances')
        for j in reversed(range(N)):
            temp = sorted(distances[j])
            lMaxTempIdxs = min(ka, len(temp))
            if lMaxTempIdxs == 0 or temp[lMaxTempIdxs] == 0:
                distances[j] = 0
            else:
                distances[j] = np.divide(distances[j], temp[lMaxTempIdxs])

    # Adjacency matrix
    print('Computing kernel')
    rows = np.zeros(N * k, dtype=np.int32)
    cols = np.zeros(N * k, dtype=np.int32)
    dists = np.zeros(N * k)
    location = 0
    for i in range(N):
        inds = range(location, location + k)
        rows[inds] = indices[i, :]
        cols[inds] = i
        dists[inds] = distances[i, :]
        location += k
    if epsilon > 0:
        W = csr_matrix((dists, (rows, cols)), shape=[N, N])
    else:
        W = csr_matrix((np.ones(dists.shape), (rows, cols)), shape=[N, N])


    # Symmetrize W
    W = W + W.T       #a是b的k个邻居，但是b可能不在a的k个邻居里


    # 高斯核函数
    # if epsilon > 0:
    #     # Convert to affinity (with selfloops)
    #     rows, cols, dists = find(W)
    #     rows = np.append(rows, range(N))
    #     cols = np.append(cols, range(N))
    #     dists = np.append(dists / (epsilon ** 2), np.zeros(N))
    #     W = csr_matrix((np.exp(-dists), (rows, cols)), shape=[N, N])


    # #t-student核函数
    if epsilon>0:
        rows, cols, dists = find(W)
        rows = np.append(rows, range(N))
        cols = np.append(cols, range(N))
        dists = np.append((1+dists)**(-2) , np.zeros(N))
        W = csr_matrix((dists, (rows, cols)), shape=[N, N])
        W = W.todense()
        sum = np.sum(W,axis=1)
        sum = np.tile(sum,(1,N))
        W = np.divide(W,sum)


    # Create D
    D = np.ravel(W.sum(axis=1))
    D[D != 0] = 1 / D[D != 0]

    # markov normalization
    T = csr_matrix((D, (range(N), range(N))), shape=[N, N]).dot(W)

    return T


def optimal_t(data, th=0.001):
    S = np.linalg.svd(data)
    S = np.power(S, 2)
    nse = np.zeros(32)

    for t in range(32):
        S_t = np.power(S, t)
        P = np.divide(S_t, np.sum(S_t, axis=0))
        nse[t] = np.sum(P[np.where(P > th)[0]])

