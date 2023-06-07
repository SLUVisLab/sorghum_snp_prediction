import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def compute_knn(vectors, query_vectors=None, self_query=True, k=10, distance_mat=False, gpu=False):
    # remove self
    index = faiss.IndexFlatL2(vectors.shape[1])
    if gpu:
        res = faiss.StandardGpuResources() 
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(vectors)
    if query_vectors is None:
        k += 1
        query_vectors = vectors
        self_query=True
    nn_distance, nn_index = index.search(query_vectors, k)
    if self_query:    
        nn_distance = nn_distance[:, 1:]
        nn_index = nn_index[:, 1:]
    if distance_mat:
        return nn_index, nn_distance
    else:
        return nn_index

def recall_at_k(vectors, labels, k=5,  gpu=True):
    if type(k) is list:
        query_k = max(k)
    else:
        query_k = k
    nn_idx = compute_knn(vectors, k=query_k, gpu=gpu)
    nn_idx = nn_idx.cpu().numpy()
    pred = labels[nn_idx[:, 0:query_k]]
    if type(k) is list:
        result = []
        for i in k:
            result.append(np.sum(np.any(labels[:,None] == pred[:,0:i],axis=1)) / (labels.shape[0]))
        return result
    else:
        return np.sum(np.any(labels[:,None] == pred,axis=1)) / (labels.shape[0])

def gallery_recall_at_k(gallery_vectors, gallery_labels, query_vectors, query_labels, k=5, gpu=True):
    if type(k) is list:
        query_k = max(k)
    else:
        query_k = k
    nn_idx = compute_knn(gallery_vectors, query_vectors, self_query=False, k=query_k, gpu=gpu)
    pred = gallery_labels[nn_idx[:, 0:query_k]]
    if type(k) is list:
        result = []
        for i in k:
            result.append(np.sum(np.any(query_labels[:,None] == pred[:, 0:i],axis=1)) / (query_labels.shape[0]))
    else:
        return np.sum(np.any(query_labels[:,None] == pred,axis=1)) / (query_labels.shape[0])

def recall_at_k_by_cultivar(vectors, labels, k=5):
    if 'cultivar_int' in labels:
        labels = labels['cultivar_int']
    elif 'cultivar' in labels:
        le = LabelEncoder()
        labels = le.fit_transform(labels['cultivar'])
    else:
        raise ValueError(f'cultivar not found in label dict')
    if type(labels) == pd.Series:
        labels = labels.values
    elif type(labels) == torch.Tensor or type(labels) == list:
        labels = np.array(labels)
    return recall_at_k(vectors, labels, k=k)

def gallery_recall_at_k_by_cultivar(vectors, labels, gallery_idx, k=5):
    if 'cultivar_int' in labels:
        labels = labels['cultivar_int']
    elif 'cultivar' in labels:
        le = LabelEncoder()
        labels = le.fit_transform(labels['cultivar'])
    else:
        raise ValueError(f'cultivar not found in label dict')
    if type(labels) == pd.Series:
        labels = labels.values
    elif type(labels) == torch.Tensor or type(labels) == list:
        labels = np.array(labels)
    g_vectors = vectors[gallery_idx]
    g_labels = labels[gallery_idx]
    query_idx = np.ones(vectors.shape[0], dtype=bool)
    query_idx[g_labels] = False
    q_vectors = vectors[query_idx]
    q_labels = labels[query_idx]
    return gallery_recall_at_k(g_vectors, g_labels, q_vectors, q_labels, k=k)

def recall_at_k_by_plot(vectors, labels, k=5):
    if 'plot_cls' in labels:
        labels = labels['plot_cls']
    elif 'plot' in labels:
        le = LabelEncoder()
        labels = le.fit_transform(labels['plot'])
    else:
        raise ValueError(f'cultivar not found in label dict')
    if type(labels) == pd.Series:
        labels = labels.values
    elif type(labels) == torch.Tensor or type(labels) == list:
        labels = np.array(labels)
    return recall_at_k(vectors, labels, k=k)

def gallery_recall_at_k_by_plot(vectors, labels, gallery_idx, k=5):
    if 'plot_cls' in labels:
        labels = labels['plot_cls']
    elif 'plot' in labels:
        le = LabelEncoder()
        labels = le.fit_transform(labels['plot'])
    else:
        raise ValueError(f'plot_cls not found in label dict')
    if type(labels) == pd.Series:
        labels = labels.values
    elif type(labels) == torch.Tensor or type(labels) == list:
        labels = np.array(labels)
    g_vectors = vectors[gallery_idx]
    g_labels = labels[gallery_idx]
    query_idx = np.ones(vectors.shape[0], dtype=bool)
    query_idx[g_labels] = False
    q_vectors = vectors[query_idx]
    q_labels = labels[query_idx]
    return gallery_recall_at_k(g_vectors, g_labels, q_vectors, q_labels, k=k)

def recall_at_k_by_date(vectors, labels, k=5):
    if 'scan_date' in labels:
        labels = labels['scan_date']
    else:
        raise ValueError(f'cultivar not found in label dict')
    if type(labels) == pd.Series:
        labels = labels.values
    return recall_at_k(vectors, labels, k=k)