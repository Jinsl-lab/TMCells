import os
import time
from typing import Mapping, Union
import psutil
import logging

import numpy as np
import anndata
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from TMCells.batch_sampler import CellSampler
from TMCells.model import TMCells
from TMCells.logging_utils import initialize_logger, log_arguments


_logger = logging.getLogger(__name__)


if not _logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    _logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class UnsupervisedTrainer:

    attr_fname: Mapping[str, str] = dict(
        model='model',
        optimizer='opt'
    )

    @log_arguments
    def __init__(self,
                 model: 'TMCells',
                 adata: anndata.AnnData,
                 init_lr: float = 5e-3,
                 lr_decay: float = 6e-5,
                 batch_size: int = 512,
                 train_instance_name: str = "TMCells",
                 seed: int = 42,
                 opt_adambetas=(0.9, 0.999),
                 ckpt_dir: Union[str, None] = None,
                 restore_epoch: int = 0
                 ) -> None:


        self.model = model
        self.train_adata = self.test_adata = self.adata = adata


        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr, betas=opt_adambetas)
        self.lr = self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.steps_per_epoch = max(self.train_adata.n_obs / self.batch_size, 1)
        self.device = model.device
        self.step = self.epoch = 0
        self.seed = seed
        self.train_instance_name = train_instance_name

        if restore_epoch > 0 and type(self) == UnsupervisedTrainer:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)
        elif ckpt_dir is not None and restore_epoch == 0:
            self.ckpt_dir = os.path.join(ckpt_dir, f"{self.train_instance_name}_{time.strftime('%m_%d-%H_%M_%S')}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            initialize_logger(self.ckpt_dir)
            _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None


    @log_arguments
    def load_ckpt(self,
                  restore_epoch: int,
                  ckpt_dir: Union[str, None] = None
                  ) -> None:


        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir
        assert ckpt_dir is not None and os.path.exists(ckpt_dir), f"ckpt_dir {ckpt_dir} does not exist."
        for attr, fname in self.attr_fname.items():
            fpath = os.path.join(ckpt_dir, f'{fname}-{restore_epoch}')
            getattr(self, attr).load_state_dict(torch.load(fpath))
        _logger.info(f'Parameters and optimizers restored from {ckpt_dir}.')
        initialize_logger(self.ckpt_dir)
        _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        self.update_step(restore_epoch * self.steps_per_epoch)


    @staticmethod
    def _calc_weight(
            epoch: int,
            t_warmup: int = 50,
            max_weight: float = 1e-4,
            min_weight: float = 0,
            cutoff_epoch: int = 0
    ) -> float:
        if epoch < cutoff_epoch:
            return min_weight

        if t_warmup <= 0:
            return max_weight
        relative_epoch = epoch - cutoff_epoch
        progress = relative_epoch / t_warmup
        
        current_weight = min_weight + (max_weight - min_weight) * progress
        return max(min(max_weight, current_weight), min_weight)



    def update_step(self, jump_to_step: Union[None, int] = None) -> None:
        if jump_to_step is None:
            self.step += 1
        else:
            self.step = jump_to_step
        self.epoch = self.step / self.steps_per_epoch
        if self.lr_decay:
            if jump_to_step is None:
                self.lr *= np.exp(-self.lr_decay)
            else:
                self.lr = self.init_lr * np.exp(-jump_to_step * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    @log_arguments
    def train(self,
              n_epochs: int = 500,
              eval_every: int = 50,
              t_warmup: float = 50,
              early_stop_epoch = 1000,
              patience = 10, 
              min_kl_weight: float = 0.,
              max_kl_weight: float = 1e-4,
              writer: Union[None, SummaryWriter] = None,
              model_dir: Union[None, str] = None,
              save_model_ckpt: bool = True,
              **train_kwargs
              ) -> None:
        sampler = CellSampler(self.train_adata, self.batch_size,
                              n_epochs=n_epochs - self.epoch,
                              rng = np.random.default_rng(seed=self.seed))

        dataloader = iter(sampler)

        next_ckpt_epoch = min(int(np.ceil(self.epoch / eval_every) * eval_every), n_epochs)

        len_epoch = 0
        loss_epoch = nll_epoch = kl_epoch = mc_nll_epoch = 0
        loss_epoch_pre = nll_epoch_pre = kl_epoch_pre = mc_nll_epoch_pre = 0
        stable_epochs = 0
        while self.epoch < n_epochs:
            new_record, hyper_param_dict = self.do_train_step(dataloader,
                                                              n_epochs=n_epochs,
                                                              t_warmup=t_warmup,
                                                              min_kl_weight=min_kl_weight,
                                                              max_kl_weight=max_kl_weight,
                                                              **train_kwargs
                                                              )

            fmt: str = "10.4g"

            pre_epoch = int(self.epoch)
            self.update_step()
 
            len_epoch += 1
            loss_epoch += new_record['loss']
            nll_epoch += new_record['nll']
            kl_epoch += new_record['kl_loss']
            mc_nll_epoch += new_record['mc_nll']

            cur_epoch = int(self.epoch)
            if cur_epoch != pre_epoch:
                loss_epoch /= len_epoch
                nll_epoch /= len_epoch
                kl_epoch /= len_epoch
                mc_nll_epoch /= len_epoch
                if cur_epoch % 10 == 0:
                    print(f'Epoch:{cur_epoch:5d}/{n_epochs:5d} loss:{loss_epoch:{fmt}} nll:{nll_epoch:{fmt}} mc_nll:{mc_nll_epoch:{fmt}} kl_loss:{kl_epoch:{fmt}} Next ckpt:{next_ckpt_epoch:5d}')

                metrics = {'loss_epoch': loss_epoch, 'nll_epoch': nll_epoch, 'kl_epoch': kl_epoch, 'mc_nll_epoch': mc_nll_epoch}
                for key, val in metrics.items():
                    if writer is not None:
                        writer.add_scalar(key, val, cur_epoch)
                if cur_epoch >= early_stop_epoch:
                    if cur_epoch > early_stop_epoch:
                        converge = (abs(loss_epoch_pre-loss_epoch) <= 1e-2)
                        if converge:
                            stable_epochs += 1
                        else:
                            stable_epochs = 0
                        if stable_epochs >= patience:
                            print("Early Stopping.")
                            break
                            
                    loss_epoch_pre = loss_epoch
                    nll_epoch_pre = nll_epoch
                    kl_epoch_pre = kl_epoch
                    mc_nll_epoch_pre = mc_nll_epoch
        
                len_epoch = 0
                loss_epoch = nll_epoch = kl_epoch = mc_nll_epoch = 0


            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {next_ckpt_epoch:.0f}' + '=' * 10)
                _logger.info(repr(psutil.Process().memory_info()))
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr:12.4g}')
                for k, v in hyper_param_dict.items():
                    _logger.info(f'{k:12s}: {v:12.4g}')

                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    self.save_model_and_optimizer(next_ckpt_epoch)

                next_ckpt_epoch = min(eval_every + next_ckpt_epoch, n_epochs)


        if model_dir is not None:
            torch.save(self.model.state_dict(), model_dir)

        _logger.info("Optimization Finished: %s" % self.ckpt_dir)

    def save_model_and_optimizer(self, next_ckpt_epoch: int) -> None:
        for attr, fname in self.attr_fname.items():
            torch.save(
                getattr(self, attr).state_dict(),
                os.path.join(self.ckpt_dir, f'{fname}-{next_ckpt_epoch}')
            )

    def do_train_step(self, dataloader, **kwargs) -> Mapping[str, torch.Tensor]:
        hyper_param_dict = {
            'kl_weight': self._calc_weight(
                self.epoch,
                kwargs['t_warmup'],
                kwargs['max_kl_weight'],
                kwargs['min_kl_weight'],
                0
            )
        }
        data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}
        new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict)

        return new_record, hyper_param_dict

