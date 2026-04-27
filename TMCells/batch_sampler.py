from typing import Iterator, Mapping, Union

import anndata
import numpy as np
import torch
import torch.sparse
from scipy.sparse import spmatrix
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class CellSampler():

    def __init__(self,
        adata: anndata.AnnData,
        batch_size: int,
        n_epochs: Union[float, int] = np.inf,
        rng: Union[None, np.random.Generator] = None,
        shuffle: bool = True
    ) -> None:

        self.n_cells: int = adata.n_obs
        self.batch_size: int = batch_size
        self.n_epochs: Union[int, float] = n_epochs
        self.is_sparse: bool = isinstance(adata.X, spmatrix)
        self.X: Union[np.ndarray, spmatrix] = adata.X

        if shuffle:
            self.rng: Union[None, np.random.Generator] = rng or np.random.default_rng()
        else:
            self.rng: Union[None, np.random.Generator] = None
        self.shuffle: bool = shuffle

        if self.is_sparse:
            self.library_size: Union[spmatrix, np.ndarray] = adata.X.sum(1) 
        else:
            self.library_size: Union[spmatrix, np.ndarray] = adata.X.sum(1, keepdims=True)

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:

        if self.batch_size < self.n_cells:
            return self._low_batch_size()
        else:
            return self._high_batch_size()

    def _high_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:

        count = 0
        X = torch.FloatTensor(self.X.todense() if self.is_sparse else self.X)
        library_size = torch.FloatTensor(self.library_size)
        cell_indices = torch.arange(0, self.n_cells, dtype=torch.long)

        result_dict = dict(cells=X, library_size=library_size, cell_indices=cell_indices)


        while count < self.n_epochs:
            count += 1
            yield result_dict


    def _low_batch_size(self) -> Iterator[Mapping[str, torch.Tensor]]:
        entry_index = 0
        count = 0
        cell_range = np.arange(self.n_cells)

        if self.shuffle:
            self.rng.shuffle(cell_range)

        while count < self.n_epochs:
            if entry_index + self.batch_size >= self.n_cells:
                count += 1
                batch = cell_range[entry_index:]
                if self.shuffle:
                    self.rng.shuffle(cell_range)
                excess = entry_index + self.batch_size - self.n_cells

                if excess > 0 and count < self.n_epochs:
                    batch = np.append(batch, cell_range[:excess], axis=0)
                    entry_index = excess
                else:
                    entry_index = 0
            else:
                batch = cell_range[entry_index: entry_index + self.batch_size]
                entry_index += self.batch_size

            library_size = torch.FloatTensor(self.library_size[batch])

            X = self.X[batch, :]
            if self.is_sparse:
                cells = torch.FloatTensor(X.todense())
            else:
                cells = torch.FloatTensor(X)
            cell_indices = torch.LongTensor(batch)

            result_dict = dict(cells=cells, library_size=library_size, cell_indices=cell_indices)
            yield result_dict





