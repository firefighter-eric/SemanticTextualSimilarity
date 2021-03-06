from dataclasses import dataclass

import torch
from pytorch_lightning import LightningModule
from scipy.stats import spearmanr
from torch.nn import CrossEntropyLoss, MSELoss, functional as F
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from model.base_model import BaseModel, Similarity, BaseModelConfig
from model.cl_loss import CLLoss
from model.tokenizer import TokenizerConfig
from utils.optim import step_warmup_optimizers


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
        self.model = BaseModel(self.config, from_pretrained)
        if self.config.mlm_flag:
            self.mlm = BertLMPredictionHead(self.model.backbone.config)
        self.sim = Similarity(self.config.temp)
        self.cl_loss = CLLoss()
        self.mlm_loss = CrossEntropyLoss()
        self.mse_loss = MSELoss()

    def forward(self, s1, s2, mlm_label_1=None, mlm_label_2=None):
        outputs_1, pooler_output_1 = self.model(**s1)
        outputs_2, pooler_output_2 = self.model(**s2)

        # sim loss
        sim = self.sim(pooler_output_1, pooler_output_2)
        sim_loss = self.cl_loss(sim)

        # mlm loss
        if self.config.mlm_flag:
            m1 = self.mlm(outputs_1)
            m2 = self.mlm(outputs_2)
            m1 = torch.permute(m1, [0, 2, 1])
            m2 = torch.permute(m2, [0, 2, 1])
            mlm_loss = self.mlm_loss(m1, mlm_label_1) + self.mlm_loss(m2, mlm_label_2)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def pair_forward(self, s1, s2, mlm_label_1=None, mlm_label_2=None, label=None):
        outputs_1, pooler_output_1 = self.model(**s1)
        outputs_2, pooler_output_2 = self.model(**s2)

        # sim loss
        sim = F.cosine_similarity(pooler_output_1, pooler_output_2)
        sim_loss = self.mse_loss(sim, label)

        # mlm loss
        if self.config.mlm_flag:
            m1 = self.mlm(outputs_1)
            m2 = self.mlm(outputs_2)
            m1 = torch.permute(m1, [0, 2, 1])
            m2 = torch.permute(m2, [0, 2, 1])
            mlm_loss = self.mlm_loss(m1, mlm_label_1) + self.mlm_loss(m2, mlm_label_2)
        else:
            mlm_loss = 0

        loss = sim_loss + self.config.mlm_weight * mlm_loss
        return sim, sim_loss, mlm_loss, loss

    def training_step(self, batch, batch_idx):
        s1, s2, mlm_label_1, mlm_label_2 = batch['s1'], batch['s2'], batch['mlm_label_1'], batch['mlm_label_2']
        sim, sim_loss, mlm_loss, loss = self(s1, s2, mlm_label_1, mlm_label_2)
        self.log('train/sim_loss', sim_loss)
        self.log('train/mlm_loss', mlm_loss)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        return self.super_validation_step(batch, batch_idx)

    def unsuper_validation_step(self, batch, batch_idx):
        s1, s2, mlm_label_1, mlm_label_2, label = \
            batch['s1'], batch['s2'], batch['mlm_label_1'], batch['mlm_label_2'], batch['label']
        sim, sim_loss, mlm_loss, loss = self(s1, s2, mlm_label_1, mlm_label_2)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        spearman_corr = spearmanr(sim, label).correlation
        self.log('val/spearman_corr', spearman_corr)

    def super_validation_step(self, batch, batch_idx):
        s1, s2, mlm_label_1, mlm_label_2, label = \
            batch['s1'], batch['s2'], batch['mlm_label_1'], batch['mlm_label_2'], batch['label']
        sim, sim_loss, mlm_loss, loss = self.pair_forward(s1, s2, mlm_label_1, mlm_label_2, label)
        self.log('val/sim_loss', sim_loss)
        self.log('val/mlm_loss', mlm_loss)
        self.log('val/loss', loss)

        sim = sim.tolist()
        label = label.tolist()
        spearman_corr = spearmanr(sim, label).correlation
        self.log('val/spearman_corr', spearman_corr)

        # batch_size = sim.size(0)
        # similarity_flatten = sim.view(-1).tolist()
        # label_flatten = torch.eye(batch_size).view(-1).tolist()
        # spearman_corr = spearmanr(similarity_flatten, label_flatten).correlation
        # self.log('val/spearman_corr', spearman_corr)
        return loss

    def configure_optimizers(self):
        return step_warmup_optimizers(self, self.config)
