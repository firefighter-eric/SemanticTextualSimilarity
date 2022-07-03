import random

import torch

DROPOUT = 0.3
stop_sign = {101, 102, 103}


def get_mlm_label(tokenized_sent):
    tokenized_sent = {k: v for k, v in tokenized_sent.items()}
    batch_size, seq_length = tokenized_sent['input_ids'].size()
    mlm_label = torch.ones(size=(batch_size, seq_length), dtype=torch.long)
    for b in range(batch_size):
        for s in range(seq_length):
            if tokenized_sent['input_ids'][b, s] not in stop_sign and random.random() < DROPOUT:
                mlm_label[b, s] = tokenized_sent['input_ids'][b, s]
                tokenized_sent['input_ids'][b, s] = 103  # [MASK]
            else:
                mlm_label[b, s] = -100
    return tokenized_sent, mlm_label
