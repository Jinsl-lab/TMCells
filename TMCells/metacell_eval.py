import anndata as ad
import pandas as pd
import numpy as np
from scipy.sparse import vstack, csr_matrix, issparse
from alive_progress import alive_bar
from scipy import sparse
import torch

def Metacell_Matrix(adata,
                    cell_type_label='cell_type',
                    metacell_label="TMCells",
                    mc_mode='average'):
    
    from collections import Counter
    
    X_original = adata.X
    
    if isinstance(X_original, np.matrix):
        print("Converting adata.X from np.matrix to np.ndarray...")
        if issparse(X_original):
            X_original = csr_matrix(X_original)
        else:
            X_original = np.array(X_original)
    
    if cell_type_label not in adata.obs:
        raise ValueError(f"'{cell_type_label}' not found in adata.obs")
    
    if metacell_label not in adata.obs:
        raise ValueError(f"'{metacell_label}' not found in adata.obs")
    
    adataX_origin = X_original
    metacell_labels = adata.obs[metacell_label].values
    cell_types = adata.obs[cell_type_label].values
    
    unique_metacells = np.unique(metacell_labels)
    
    out = []
    mc_celltypes = []
    mc_purities = []
    
    for metacell in unique_metacells:
        cell_mask = (metacell_labels == metacell)
        cell_indices = np.where(cell_mask)[0]
        
        if len(cell_indices) == 0:
            continue
        
        cell_types_in_mc = cell_types[cell_indices]
        type_counter = Counter(cell_types_in_mc)
        most_common_type, most_common_count = type_counter.most_common(1)[0]
        
        purity = most_common_count / len(cell_indices)
        
        mc_celltypes.append(most_common_type)
        mc_purities.append(purity)
        
        subset = adataX_origin[cell_indices, :]
        
        if isinstance(subset, np.matrix):
            subset = subset.A
        
        if mc_mode == 'average':
            if hasattr(subset, 'toarray'):
                avg = subset.mean(axis=0)
                if avg.ndim == 1:
                    avg = avg.reshape(1, -1)
            else:
                avg = np.mean(subset, axis=0, keepdims=True)
            out.append(avg)
            
        elif mc_mode == 'sum':
            if hasattr(subset, 'toarray'):
                total = subset.sum(axis=0)
                if total.ndim == 1:
                    total = total.reshape(1, -1)
            else:
                total = np.sum(subset, axis=0, keepdims=True)
            out.append(total)
        else:
            raise ValueError("mc_mode must be either 'average' or 'sum'")
    
    if out:
        out_2d = [x if x.ndim == 2 else x.reshape(1, -1) for x in out]
        
        if all(hasattr(x, 'toarray') for x in out_2d):
            X_combined = vstack(out_2d)
        else:
            X_combined = np.vstack(out_2d)
        
        mc_adata = ad.AnnData(
            X=X_combined,
            obs=pd.DataFrame(
                index=unique_metacells.astype(str),
                data={
                    'cell_type': mc_celltypes,
                    'purity': mc_purities
                }
            ),
            var=adata.var.copy()
        )
    else:
        mc_adata = ad.AnnData(
            X=csr_matrix((0, adata.shape[1])),
            obs=pd.DataFrame(index=[]),
            var=adata.var.copy()
        )
    
    mc_adata.var_names = adata.var_names.copy()
    
    return mc_adata




def celltype_frac(x, col_name):
    val_counts = x[col_name].value_counts()
    return val_counts.values[0] / val_counts.values.sum()


def compute_celltype_purity(ad, celltype_label, metacell_label="metacell"):
    celltype_fraction = ad.obs.groupby(metacell_label).apply(
        lambda x: celltype_frac(x, celltype_label)
    )

    celltype = ad.obs.groupby(metacell_label).apply(
        lambda x: x[celltype_label].value_counts().index[0]
    )
    return pd.concat([celltype, celltype_fraction], axis=1).rename(
        columns={0: celltype_label, 1: f"{celltype_label}_purity"}
    )

def pearson_pytorch(A, B):
    if isinstance(A, sparse.csr_matrix) or isinstance(A, sparse.csc_matrix):
        A = A.toarray()
    if isinstance(B, sparse.csr_matrix) or isinstance(B, sparse.csc_matrix):
        B = B.toarray()
    A = torch.from_numpy(A).half()
    B = torch.from_numpy(B).half()
    if torch.cuda.is_available():
        A = A.cuda()
        B = B.cuda()

    A_mean = A - A.mean(dim=1, keepdim=True)
    B_mean = B - B.mean(dim=1, keepdim=True)
    A_std = A_mean.norm(dim=1)
    B_std = B_mean.norm(dim=1)
    cov_matrix = torch.mm(A_mean, B_mean.t())
    correlation_matrix = cov_matrix / torch.outer(A_std, B_std)

    return correlation_matrix.cpu().numpy()


def pairwise_correlation(data):
    cell_num = data.shape[0]
    print("Computing Pairwise Correlation...")
    corr = np.zeros((cell_num, cell_num), dtype=np.float16)
    chunk_size = 5000
    chunks_num = cell_num // chunk_size + (1 if cell_num % chunk_size != 0 else 0)
    total_steps = chunks_num * chunks_num

    with alive_bar(total_steps, enrich_print=False) as bar:
        for i in range(0, cell_num, chunk_size):
            row_start, row_end = i, min(i + chunk_size, cell_num)
            for j in range(0, cell_num, chunk_size):
                col_start, col_end = j, min(j + chunk_size, cell_num)
                corr[row_start:row_end, col_start:col_end] = pearson_pytorch(
                    data[row_start:row_end], data[col_start:col_end]
                )
                bar() 
    return corr

def compute_compactness(corr, metacell_label):
    cell_num = corr.shape[0]
    metacell_num = len(np.unique(metacell_label))
    assignment_ids = np.unique(metacell_label)
    compactness = []
    for i in assignment_ids:
        idx = np.where(metacell_label == i)[0]
        if len(idx) == 0:
            continue
        compactness.append(
            np.mean(corr[idx][:, idx]) * len(idx) / cell_num * metacell_num
        )
    return compactness


def compute_separation(corr, metacell_label):
    cell_num = corr.shape[0]
    metacell_num = len(np.unique(metacell_label))
    assignment_ids = np.unique(metacell_label)
    separation = []
    for i in assignment_ids:
        idx = np.where(metacell_label == i)[0]
        complementary_idx = np.where(metacell_label != i)[0]
        if len(idx) == 0:
            continue
        separation.append(
            np.mean(1 - corr[idx][:, complementary_idx].max(axis=1))
            * len(idx)
            / cell_num
            * metacell_num
        )
    return separation