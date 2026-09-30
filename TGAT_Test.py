from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from sklearn.linear_model import LogisticRegression
from torch import Tensor
import numpy as np
import torch
import dgl
#import TCA
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_mask(g, critical_nodes):
    node_types = g.ndata['feat']
    critical_nodes_tensor = torch.tensor(critical_nodes, device=node_types.device)
    mask = torch.isin(node_types, critical_nodes_tensor).float()
    return mask.to(DEVICE)

def WGNN_test(critical_nodes, test_tar, model, train_graph, test_graph, train_label, test_label, acc, auc, f1, mcc, gmean, precision, recall):

    alpha = 0
    train_graph_mmd = dgl.batch(train_graph)
    test_graph_mmd = dgl.batch(test_graph)

    Train_mask = generate_mask(train_graph_mmd, critical_nodes)
    Test_mask = generate_mask(test_graph_mmd, critical_nodes)

    model.eval()
    with torch.no_grad():
        train_x, _, _ ,_ , _, = model(train_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Train_mask, compute_node_mmd=False)
        test_x, _, _ ,_ , _, = model(test_graph_mmd, train_graph_mmd, test_graph_mmd, alpha, mask=Test_mask, compute_node_mmd=False)

    if type(train_x) is Tensor:
        train_x = train_x.data.cpu().numpy()
        test_x = test_x.data.cpu().numpy()

    if type(train_label) is Tensor:
        train_label = train_label.data.cpu().numpy()
        test_label = test_label.data.cpu().numpy()

    # Z-score
    train_x = (train_x - np.mean(train_x, axis=0)) / np.std(train_x, axis=0)
    test_x = (test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis=0)

    pca = PCA(n_components = 0.3)
    pca.fit(train_x)
    train_x_pca = pca.transform(train_x)
    test_x_pca = pca.transform(test_x)

    # Perform Prediction
    cls = LogisticRegression(max_iter=1000)
    cls.fit(train_x_pca, train_label.ravel())
    y_pred = cls.predict(test_x_pca)
    y_proba = cls.predict_proba(test_x_pca)[:, 1]


    train_label = np.asarray(train_label).ravel()
    test_label = np.asarray(test_label).ravel()

    # Save Result
    acc.append(accuracy_score(y_true=test_label.ravel(), y_pred=y_pred))
    auc.append(roc_auc_score(y_true=test_label.ravel(), y_score=y_proba))
    f1.append(f1_score(y_true=test_label.ravel(), y_pred=y_pred))
    mcc.append(matthews_corrcoef(y_true=test_label.ravel(), y_pred=y_pred))
    gmean.append(geometric_mean_score(y_true=test_label.ravel(), y_pred=y_pred))
    precision.append(precision_score(y_true=test_label.ravel(), y_pred=y_pred))
    recall.append(recall_score(y_true=test_label.ravel(), y_pred=y_pred))

    return acc, auc, f1, mcc, gmean, precision, recall



