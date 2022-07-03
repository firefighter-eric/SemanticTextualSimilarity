from dataclasses import dataclass

from transformers import AutoTokenizer


@dataclass
class TokenizerConfig:
    tokenizer_name: str = 'bert-base-cased'


class Tokenizer:
    def __init__(self, config:TokenizerConfig):
        self.t = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def __call__(self, sents):
        return self.t(sents, return_tensors='pt', padding=True, truncation=True)
