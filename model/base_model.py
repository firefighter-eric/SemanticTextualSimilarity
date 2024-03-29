from dataclasses import dataclass

from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        x = F.normalize(x)
        y = F.normalize(y)
        return x @ y.T / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def get_model(model_name, from_pretrained):
    if from_pretrained:
        model = AutoModel.from_pretrained(model_name)
    else:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_config(config)
    return model


class BaseModelPooler(nn.Module):
    def __init__(self, config, from_pretrained):
        super().__init__()
        self.config = config
        self.backbone = get_model(config.model_name, from_pretrained)
        self.pooler = Pooler(config.pool_type)

    def forward(self, x):
        outputs = self.backbone(x)
        pooler_output = self.pooler(outputs)
        return outputs, pooler_output


@dataclass
class BaseModelConfig:
    model_name: str = 'bert-base-uncased'


class BaseModel(nn.Module):
    def __init__(self, config, from_pretrained):
        super().__init__()
        self.config = config
        self.backbone = get_model(config.model_name, from_pretrained)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = outputs[:, 0]
        return outputs, pooler_output


class ModelWithPooler(BaseModel):
    def __init__(self, config, from_pretrained):
        super().__init__(config, from_pretrained)
        self.mlp = MLPLayer(self.backbone.config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(input_ids, attention_mask, token_type_ids)[0]
        pooler_output = self.mlp(outputs[:, 0])
        return outputs, pooler_output
