from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class TokenizerConfig:
    tokenizer_name: str = 'bert-base-cased'
    model_max_length: int = 512


class Tokenizer:
    def __init__(self, config: TokenizerConfig):
        self.t = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.t.model_max_length = config.model_max_length

    def __call__(self, sents):
        return self.t(sents, return_tensors='pt', padding=True, truncation=True)
