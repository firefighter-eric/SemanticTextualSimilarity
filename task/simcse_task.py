from dataclasses import dataclass

import numpy as np
import torch
from pytorch_lightning import LightningModule
from scipy.stats import spearmanr
from torch.nn import CrossEntropyLoss, MSELoss, functional as F
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from model.base_model import Similarity, BaseModelConfig, ModelWithPooler
from model.cl_loss import CLLoss
from model.tokenizer import TokenizerConfig
from utils.optim import get_linear_schedule_with_warmup


@dataclass
class SimCSETaskConfig(BaseModelConfig, TokenizerConfig):
    mlm_flag: bool = True
    mlm_weight: float = 0.1

    temp: float = 0.05

    lr: float = 1e-5
    weight_decay: float = 0
    warnup_steps: int = 1000


class SimCSETask(LightningModule):
    def __init__(self, config: SimCSETaskConfig, from_pretrained=False):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore='from_pretrained')
        self.model = ModelWithPooler(self.config, from_pretrained)
        if self.config.mlm_flag:
            self.mlm = BertLMPredictionHead(self.model.backbone.config)
        self.sim = Similarity(self.config.temp)
        self.cl_loss = CLLoss()
        self.mlm_loss = CrossEntropyLoss()
        self.mse_loss = MSELoss()

    def forward(self, sent, mlm_label=None):
        outputs, pooler_output = self.model(**sent)
        pooler_output = pooler_output.view(pooler_output.size(0) // 2, 2, -1)
        z1 = pooler_output[:, 0]
        z2 = pooler_output[:, 1]

        # sim loss
        sim = self.sim(z1, z2)
        sim_loss = self.cl_loss(sim)

        # mlm loss
        if self.config.mlm_flag:
            mlm_outputs = self.mlm(outputs)
            m1 = torch.permute(mlm_outputs, [0, 2, 1])
            mlm_loss = self.mlm_loss(m1, mlm_label)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def pair_forward(self, sent, mlm_label, label=None):
        outputs, pooler_output = self.model(**sent)
        pooler_output = pooler_output.view(pooler_output.size(0) // 2, 2, -1)
        z1 = pooler_output[:, 0]
        z2 = pooler_output[:, 1]

        # sim loss
        sim = F.cosine_similarity(z1, z2)
        sim_loss = self.mse_loss(sim, label)

        # mlm loss
        if self.config.mlm_flag:
            mlm_outputs = self.mlm(outputs)
            mlm_outputs = torch.permute(mlm_outputs, [0, 2, 1])
            mlm_loss = self.mlm_loss(mlm_outputs, mlm_label)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def training_step(self, batch, batch_idx):
        sent, mlm_label = batch['sent'], batch['mlm_label']
        sim, sim_loss, mlm_loss, loss = self(sent, mlm_label)
        self.log('train/sim_loss', sim_loss)
        self.log('train/mlm_loss', mlm_loss)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        return self.super_validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs) -> None:
        sim, label = zip(*outputs)
        sim = torch.concat(sim).tolist()
        label = torch.concat(label).tolist()
        spearman_corr = spearmanr(sim, label)[0]
        self.log('val/spearman_corr', spearman_corr)

    def unsuper_validation_step(self, batch, batch_idx):
        sent, mlm_label, label = batch['sent'], batch['mlm_label'], batch['label']
        sim, sim_loss, mlm_loss, loss = self(sent, mlm_label)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        spearman_corr = spearmanr(sim, label).correlation
        self.log('val/spearman_corr', spearman_corr)

    def super_validation_step(self, batch, batch_idx):
        sent, mlm_label, label = batch['sent'], batch['mlm_label'], batch['label']
        sim, sim_loss, mlm_loss, loss = self.pair_forward(sent, mlm_label, label)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)
        return sim, label

    def configure_optimizers(self):
        return get_linear_schedule_with_warmup(self, self.config, self.trainer)
