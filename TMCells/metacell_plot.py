import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import sparse
import scanpy as sc
from typing import List, Optional, Dict


def plot_metacell_2D(
    ad,
    metacell_index="TMCells",
    key="X_umap",
    colour_metacells=True,
    title="TMCells",
    save_as=None,
    show=True,
    cmap="Set2",
    figsize=(5, 5),
    SEACell_size=20,
    cell_size=10,
    ax=None
):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    umap = pd.DataFrame(ad.obsm[key]).set_index(ad.obs_names).join(ad.obs[metacell_index])
    umap[metacell_index] = umap[metacell_index].astype("category")
    mcs = umap.groupby(metacell_index).mean().reset_index()

    if colour_metacells:
        sns.scatterplot(x=0, y=1, hue=metacell_index, data=umap, s=cell_size,
                        cmap=cmap, legend=None, ax=ax)
        sns.scatterplot(x=0, y=1, s=SEACell_size, hue=metacell_index, data=mcs,
                        cmap=cmap, edgecolor="black", linewidth=1.25, legend=None, ax=ax)
    else:
        sns.scatterplot(x=0, y=1, color="grey", data=umap, s=cell_size, ax=ax)
        sns.scatterplot(x=0, y=1, s=SEACell_size, color="red", data=mcs,
                        edgecolor="black", linewidth=1.25, ax=ax)

    ax.set_xlabel(f"{key}-0")
    ax.set_ylabel(f"{key}-1")
    ax.set_title(title)
    ax.set_axis_off()

    if save_as is not None:
        plt.savefig(save_as, dpi=150, transparent=True)
    if show and ax is None:  
        plt.show()
        plt.close()


def plot_metacell_2D_bycelltype(
        ad,
        metacell_index="TMCells",
        celltype_key="predicted.id",
        key="X_umap",
        colour_metacells=True,
        title="TMCells",
        save_as=None,
        show=True,
        cmap="tab20",
        figsize=(5, 5),
        metacell_size=15,
        cell_size=10,
        ax=None
):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    umap = pd.DataFrame(ad.obsm[key], index=ad.obs_names).join(
        ad.obs[[metacell_index, celltype_key]]
    ).reset_index(drop=True).rename(columns={0: f"{key}_0", 1: f"{key}_1"})

    metacell_celltypes = ad.obs.groupby(metacell_index).apply(
        lambda x: x[celltype_key].value_counts().index[0]
    ).reset_index().rename(columns={0: celltype_key})

    mcs = umap.groupby(metacell_index).agg({f"{key}_0": 'mean', f"{key}_1": 'mean'}).reset_index()
    mcs = mcs.merge(metacell_celltypes, on=metacell_index)
    unique_celltypes = sorted(umap[celltype_key].unique())

    if isinstance(cmap, str) and cmap == "tab20":
        cmap = plt.cm.get_cmap("tab20")
        colors = [cmap(i) for i in range(min(len(unique_celltypes), 20))]
        if len(unique_celltypes) > 20:
            cmap_extended = plt.cm.get_cmap("tab20b")
            colors.extend([cmap_extended(i) for i in range(min(len(unique_celltypes) - 20, 20))])

            if len(unique_celltypes) > 40:
                cmap_extended = plt.cm.get_cmap("tab20c")
                colors.extend([cmap_extended(i) for i in range(len(unique_celltypes) - 40)])
    else:
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
            colors = [cmap(i % cmap.N) for i in range(len(unique_celltypes))]
        else:
            colors = cmap

    color_dict = {celltype: color for celltype, color in zip(unique_celltypes, colors)}

    if colour_metacells:
        sns.scatterplot(
            x=f"{key}_0", y=f"{key}_1",
            hue=celltype_key,
            palette=color_dict,
            data=umap,
            s=cell_size,
            legend=False,
            ax=ax
        )

        p = sns.scatterplot(
            x=f"{key}_0", y=f"{key}_1",
            hue=celltype_key,
            palette=color_dict,
            s=metacell_size,
            data=mcs,
            edgecolor="black",
            linewidth=1.25,
            # legend="brief",
            legend=False,
            ax=ax
        )
        #p.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        

    else:
        sns.scatterplot(
            x=f"{key}_0", y=f"{key}_1",
            color="grey",
            data=umap,
            s=cell_size,
            ax=ax
        )
        sns.scatterplot(
            x=f"{key}_0", y=f"{key}_1",
            s=SEACell_size,
            color="red",
            data=mcs,
            edgecolor="black",
            linewidth=1.25,
            ax=ax
        )

    ax.set_xlabel(f"{key}_0")
    ax.set_ylabel(f"{key}_1")
    ax.set_title(title, fontsize=20)
    ax.set_axis_off()
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
    if show and ax is None:
        plt.show()
        plt.close()

    return color_dict


def plot_metacell_sizes(
    ad,
    metacell_index = "TMCells",
    save_as=None,
    show=True,
    title="Distribution of Metacell Sizes",
    bins=None,
    figsize=(5, 5),
    xlim=None,
    ylim=None,
):

    assert metacell_index in ad.obs, 'AnnData must contain "TMCells" in obs DataFrame.'
    label_df = ad.obs[[metacell_index]].reset_index()
    plt.figure(figsize=figsize)
    sns.distplot(label_df.groupby(metacell_index).count().iloc[:, 0], bins=bins)
    sns.despine()
    plt.xlabel("Number of Cells per MetaCell")
    plt.title(title)
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    plt.grid(False)
    if save_as is not None:
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    return pd.DataFrame(label_df.groupby(metacell_index).count().iloc[:, 0]).rename(
        columns={"index": "size"}
    )

def plot_metacell_gene_activity(
        adata,
        celltype_df: pd.DataFrame,
        x_gene: str,
        y_gene: str,
        target_celltypes: List[str],
        color_dict: Optional[Dict[str, str]] = None,
        figsize: tuple = (10, 6),
        point_size: int = 60,
        font_size: int = 12,
        tick_fontsize: int = 12,
        alpha: float = 0.9,
        save_path: Optional[str] = None,
        dpi: int = 300,
        adjust: float = 0.1,
        **kwargs
):

    if x_gene not in adata.var_names:
        raise ValueError(f"Gene '{x_gene}' not found in adata.var_names")
    if y_gene not in adata.var_names:
        raise ValueError(f"Gene '{y_gene}' not found in adata.var_names")

    mask = celltype_df['predicted.id'].isin(target_celltypes)
    if not mask.any():
        raise ValueError("No metacells found for the specified cell types")

    filtered_metacells = adata[mask]
    filtered_celltypes = celltype_df.loc[mask, 'predicted.id'].to_numpy()

    x_idx = np.where(adata.var_names == x_gene)[0][0]
    y_idx = np.where(adata.var_names == y_gene)[0][0]

    x_activity = filtered_metacells.X[:, x_idx].toarray().flatten()
    y_activity = filtered_metacells.X[:, y_idx].toarray().flatten()

    if color_dict is not None:
        palette = {ct: color_dict[ct] for ct in target_celltypes if ct in color_dict}
        missing_colors = set(target_celltypes) - set(palette.keys())
        if missing_colors:
            print(f"Warning: Missing colors for cell types: {missing_colors}")
            default_colors = sns.color_palette("tab10", n_colors=len(missing_colors))
            for i, ct in enumerate(missing_colors):
                palette[ct] = default_colors[i]
    else:
        default_colors = sns.color_palette("tab10", n_colors=len(target_celltypes))
        palette = {celltype: color for celltype, color in zip(target_celltypes, default_colors)}

    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(
        x=x_activity,
        y=y_activity,
        hue=filtered_celltypes,
        palette=palette,
        s=point_size,
        alpha=alpha,
        edgecolor='black',
        linewidth=0.5,
        **kwargs
    )

    plt.xlim(left=min(x_activity) - adjust, right=max(x_activity) + adjust)
    plt.ylim(bottom=min(y_activity) - adjust, top=max(y_activity) + adjust)

    plt.xlabel(f'{x_gene}', fontsize=font_size)
    plt.ylabel(f'{y_gene}', fontsize=font_size)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(
        title='Cell Type',
        title_fontsize=font_size,
        fontsize=font_size,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        frameon=True,
        shadow=True
    )

    plt.tight_layout()
    plt.grid(False)
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()




def create_normalized_adata(original_adata, normalization_method="cpm"):

    new_adata = original_adata.copy()
    X = original_adata.X
    if normalization_method == "log1p":
        if sparse.issparse(X):
            new_adata.X = X.log1p()
        else:
            new_adata.X = np.log1p(X)

    elif normalization_method == "cpm":
        if sparse.issparse(X):
            X = X.astype(np.float32)
            counts = np.array(X.sum(axis=1)).flatten()
            counts[counts == 0] = 1
            scale = 1e6 / counts
            new_adata.X = sparse.diags(scale) @ X
        else:
            X = X.astype(np.float32)
            counts = X.sum(axis=1, keepdims=True)
            counts[counts == 0] = 1
            new_adata.X = X / counts * 1e6

    elif normalization_method == "pf":

        new_adata = original_adata.copy()
        sc.pp.normalize_total(new_adata, target_sum=1e4)

    else:
        raise ValueError(f"Unsupported normalization method: {normalization_method}")

    new_adata.layers["raw"] = original_adata.X.copy()

    return new_adata


