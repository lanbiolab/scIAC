from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scIAC import scDCC

import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from utils import cluster_acc, generate_random_pair
import pandas as pd
from impute_jaccard import impute_jaccard
import magic_raw
import stimpute
from sklearn.feature_extraction.text import TfidfTransformer


if __name__ == "__main__":

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--data_file', default=r'')
    parser.add_argument('--label_cells_files', default=r'')
    parser.add_argument('--n_pairwise', default=5000, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=300, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--ae_weight_file', default=r'')
    parser.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parser.add_argument("--highly_genes", default = 1000)
    parser.add_argument('--outdir', '-o', type=str, default='results/tsne/hek/', help='Output path')
    args = parser.parse_args()

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'
    print(device)

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()


    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    # magic_operator = MAGIC()
    # adata.X = magic_operator.fit_transform(adata.X)


    # adata.X[adata.X > 0] = 1


    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)



    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    # adata = preprocess(adata,
    #                   copy=True,
    #                   highly_genes=args.highly_genes,
    #                   size_factors=True,
    #                   normalize_input=True,
    #                   logtrans_input=True)

    adata.X = stimpute.stimpute(adata.X, n_components=8, random_pca=True, t=3, k=15, ka=5, epsilon=1, rescale=99)
    print(adata.X.shape)
    input_size = adata.n_vars


    # adata.X = impute_jaccard(adata.X)

    # adata.X = np.loadtxt(r'C:\Users\23207\Desktop\scDCC-master\data\atac_seq\Forebrain\TF-IDFTXT')

    adata.X = adata.X.astype(np.float32)

    # adata.raw = adata.copy()

    print(adata.X)
    # np.savetxt(r'C:\Users\23207\Desktop\scDCC-master\data\magic_deal_data\forebrain\imputed_data.txt',adata.X)
    print(args)


    print(adata.X.shape)
    print(y.shape)

    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells * len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    if args.n_pairwise > 0:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num = generate_random_pair(y, label_cell_indx, args.n_pairwise,
                                                                             args.n_pairwise_error)

        print("Must link paris: %d" % ml_ind1.shape[0])
        print("Cannot link paris: %d" % cl_ind1.shape[0])
        print("Number of error pairs: %d" % error_num)
    else:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])

    sd = 2.5

    #leuk
    model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters,
                  encodeLayer=[512,128], decodeLayer=[128,512], sigma=sd, gamma=args.gamma,
                  ml_weight=args.ml_weight, cl_weight=args.ml_weight).cuda()
    # model = scDCC(input_dim=adata.n_vars, z_dim=64, n_clusters=args.n_clusters,
    #               encodeLayer=[512,256], decodeLayer=[256,512], sigma=sd, gamma=args.gamma,
    #               ml_weight=args.ml_weight, cl_weight=args.ml_weight).cuda()
    # model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters,
    #           encodeLayer=[256,128], decodeLayer=[128,256], sigma=sd, gamma=args.gamma,
    #               ml_weight=args.ml_weight, cl_weight=args.ml_weight).cuda()


    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors,
                                   batch_size=args.batch_size, epochs=args.pretrain_epochs,
                                   ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError

    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=y,
                                   batch_size=args.batch_size, num_epochs=args.maxiter,
                                   ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                                   update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))

    eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
    eval_cell_y = np.delete(y, label_cell_indx)
    acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    if not os.path.exists(args.label_cells_files):
        np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")

    outdir = args.outdir+'/'
    sc.settings.figdir = outdir
    latent = model.encodeBatch(X=torch.tensor(adata.X).cuda())
    adata.obsm['latent'] = latent.cpu().numpy()



    # sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    # sc.tl.leiden(adata)
    # label = np.array(adata.obs['leiden'].astype(int))
    # true_label = np.loadtxt(r'C:\Users\23207\Desktop\scDCC-master\data\atac_seq\HEK\label__HEK.txt')
    # true_label = np.array(true_label)
    # acc = np.round(cluster_acc(true_label, label), 5)
    # nmi = np.round(metrics.normalized_mutual_info_score(true_label, label), 5)
    # ari = np.round(metrics.adjusted_rand_score(true_label, label), 5)
    #
    # print('test ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    sc.set_figure_params(dpi=100, figsize=(6,6), fontsize=10)
    sc.tl.tsne(adata, use_rep='latent')
    # color = [c for c in ['celltype', args.cluster_method] if c in adata.obs]
    adata.obs['SCIAC'] = y_pred
    sc.pl.tsne(adata, color=['SCIAC'], save='.pdf', wspace=0.4, ncols=4)