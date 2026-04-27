from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple, Union
import anndata
import numpy as np
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from TMCells.batch_sampler import CellSampler
from scipy.optimize import linear_sum_assignment
from .model_utils import set_seed

_logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TMCells(nn.Module):
    clustering_input: str = "delta"
    emb_names: Sequence[str] = ['delta', 'theta']
    predict_names: str = "TMCells"
    max_logsigma = 10
    min_logsigma = -10

    def __init__(self,
                 n_features: int,
                 n_metacells: int,
                 n_topics: int = 15,
                 trainable_gene_emb_dim: int = 400,
                 hidden_sizes: list[int] = [128],
                 bn: bool = True,
                 dropout_prob: float = 0.1,
                 normalize_beta: bool = False,
                 normed_loss: bool = True,
                 norm_cells: bool = True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 seed = 42
                 ):

        if seed >= 0:
            set_seed(seed)

        super().__init__()
        self.n_features: int = n_features
        self.n_metacells: int = n_metacells

        self.n_topics: int = n_topics
        self.trainable_gene_emb_dim: int = trainable_gene_emb_dim
        self.hidden_sizes: Sequence[int] = hidden_sizes
        self.bn: bool = bn
        self.dropout_prob: float = dropout_prob
        self.normalize_beta: True = normalize_beta
        self.normed_loss: bool = normed_loss
        self.norm_cells: bool = norm_cells
        self.device: torch.device = device

        encoder_layers = []
        prev_dim = n_features

        for h_dim in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())

            if bn:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            if dropout_prob:
                encoder_layers.append(nn.Dropout(dropout_prob))

            prev_dim = h_dim

        self.q_delta = nn.Sequential(*encoder_layers)

        hidden_dim = hidden_sizes[-1]

        self.mu_q_delta: nn.Linear = nn.Linear(hidden_dim, n_topics, bias=True)
        self.logsigma_q_delta: nn.Linear = nn.Linear(hidden_dim, n_topics, bias=True)
        self.rho: nn.Parameter = nn.Parameter(torch.randn(self.trainable_gene_emb_dim, self.n_features))
        self.alpha: nn.Parameter = nn.Parameter(torch.randn(self.n_topics, self.trainable_gene_emb_dim))

        self.pi_ = nn.Parameter(torch.FloatTensor(self.n_metacells, ).fill_(1) / self.n_metacells, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(self.n_metacells, self.n_topics).fill_(0),requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(self.n_metacells, self.n_topics).fill_(0),requires_grad=True)

        self.to(device)



    def decode(self,
               theta: torch.Tensor
               ) -> torch.Tensor:

        beta = self.alpha @ self.rho

        if self.normalize_beta:
            recon = torch.mm(theta, F.softmax(beta, dim=-1))
            recon_log = (recon + 1e-30).log()
        else:
            recon_logit = torch.mm(theta, beta)
            recon_log = F.log_softmax(recon_logit, dim=-1)
        return recon_log

    def forward(self,
                data_dict: Mapping[str, torch.Tensor],
                hyper_param_dict: Mapping[str, Any] = dict()
                ) -> Mapping[str, Any]:

        cells, library_size = data_dict['cells'], data_dict['library_size']
        normed_cells = cells / library_size
        input_cells = normed_cells if self.norm_cells else cells

        q_delta = self.q_delta(input_cells)
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta).clamp(self.min_logsigma, self.max_logsigma)


        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta.rsample()
        theta = F.softmax(delta, dim=-1)

        if not self.training:
            theta = F.softmax(mu_q_delta, dim=-1)
            predict,yita = self.predict(mu_q_delta, is_print = True)
            fwd_dict = dict(theta=theta, delta=mu_q_delta, TMCells=predict)
            return fwd_dict

        loss = 0

        recon_log = self.decode(theta)
        nll = (-recon_log * normed_cells if self.normed_loss else cells).sum(-1).mean()
        loss += nll

        predict,yita = self.predict(delta)
        predict_tensor = torch.from_numpy(predict).squeeze().long()
        selected_mu = self.mu_c[predict_tensor]
        selected_log_sigma_c = 0.5 * self.log_sigma2_c[predict_tensor]
        selected_q_delta = Independent(Normal(
            loc=selected_mu,
            scale=selected_log_sigma_c.exp()
        ), 1)
        selected_delta = selected_q_delta.rsample()
        selected_theta = F.softmax(selected_delta, dim=-1)
        mc_recon_log = self.decode(selected_theta)
        mc_nll = (-mc_recon_log * normed_cells if self.normed_loss else cells).sum(-1).mean()
        loss += mc_nll

        kl_weight = hyper_param_dict['kl_weight']
        kl_loss = self.ELBO_kl_Loss(mu_q_delta, logsigma_q_delta)
        loss += kl_weight * kl_loss

        record = dict(loss=loss, nll=nll, kl_loss=kl_loss, mc_nll=mc_nll)
        record = {k: v.detach().item() for k, v in record.items()}

        fwd_dict = dict(
            theta=theta,
            delta=delta,
            recon_log=recon_log
        )

        return loss, fwd_dict, record


    def train_step(self,
                   optimizer: optim.Optimizer,
                   data_dict: Mapping[str, torch.Tensor],
                   hyper_param_dict: Mapping[str, Any],
                   ) -> Mapping[str, torch.Tensor]:

        self.train()
        optimizer.zero_grad()
        loss, fwd_dict, new_record = self(data_dict, hyper_param_dict)
        loss.backward()
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        new_record['max_norm'] = norms.cpu().numpy()
        optimizer.step()

        return new_record


    def ELBO_kl_Loss(self, mu_q_delta, logsigma_q_delta):

        det = 1e-10
        kl_loss = 0

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        q_delta = Independent(Normal(
            loc=mu_q_delta,
            scale=logsigma_q_delta.exp()
        ), 1)

        delta = q_delta.rsample()

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(delta, mu_c, log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))

        logsigma2_q_delta = 2 * logsigma_q_delta
        kl_loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0) +
                                                              torch.exp(logsigma2_q_delta.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                              (mu_q_delta.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2),1)
                                 )

        kl_loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1))
        kl_loss -= 0.5 * torch.mean(torch.sum(1 + logsigma2_q_delta, 1))

        return kl_loss


    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.n_metacells):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def _apply_to(self,
                  adata: anndata.AnnData,
                  batch_size: int = 512,
                  hyper_param_dict: Union[dict, None] = None,
                  callback: Union[Callable, None] = None
                  ) -> None:

        sampler = CellSampler(adata, batch_size=batch_size, n_epochs=1, shuffle=False)
        self.eval()

        for data_dict in sampler:
            data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
            fwd_dict = self(data_dict, hyper_param_dict=hyper_param_dict)
            if callback is not None:
                callback(data_dict, fwd_dict)

    def get_cell_embeddings_and_metacells(self,
                                   adata: anndata.AnnData,
                                   batch_size: int = 2000,
                                   emb_names: Union[str, Iterable[str], None] = None,
                                   inplace: bool = True   
                                   ) -> Union[Union[None, float], Tuple[Mapping[str, np.ndarray], Union[None, float]]]:

        assert adata.n_vars == self.n_features

        if emb_names is None:
            emb_names = self.emb_names
        self.eval()

        if isinstance(emb_names, str):
            emb_names = [emb_names]

        embs = {name: [] for name in emb_names}
        predict = []

        hyper_param_dict = {}

        def store_emb_and_metacell(data_dict, fwd_dict):
            predict.append(fwd_dict["TMCells"])
            for name in emb_names:
                embs[name].append(fwd_dict[name].detach().cpu())

        self._apply_to(adata, batch_size, hyper_param_dict, callback=store_emb_and_metacell)

        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}

        if inplace:
            adata.obsm.update(embs)
            all_predict = np.concatenate(predict, axis=0)
            adata.obs = adata.obs.assign(TMCells=all_predict)
            if adata.is_view:
                adata = adata.copy()
            adata.varm['rho'] = self.rho.T.detach().cpu().numpy()

            adata.uns['alpha'] = self.alpha.detach().cpu().numpy()
            return
        else:
            result_dict = embs
            if adata.is_view:
                adata = adata.copy()
            adata.varm['rho'] = self.rho.T.detach().cpu().numpy()
            result_dict['alpha'] = self.alpha.detach().cpu().numpy()
            return

    def predict(self, delta, is_print = False):
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(delta, mu_c, log_sigma2_c))
        yita = yita_c.detach().cpu().numpy()
        predict = np.argmax(yita, axis=1)
        if is_print:
            print(f"predict:{predict},len:{len(predict)}")
        return predict, yita


