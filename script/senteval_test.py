from pprint import pprint
from typing import Dict

import torch
from transformers import AutoTokenizer

import senteval
from task.simcse_task import SimCSETask

PATH_TO_DATA = '../data'


def evaluate(model, tokenizer, device) -> Dict[str, float]:
    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(sentences,
                                            return_tensors='pt',
                                            padding=True, )
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
            pooler_output = outputs[1]
        return pooler_output.cpu()

    # Set params for SentEval (fast mode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5,
              'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                             'tenacity': 3, 'epoch_size': 2}}

    print('Initializing SentEval...')
    se = senteval.engine.SE(params, batcher, prepare)
    # tasks = ['STSBenchmark', 'SICKRelatedness']
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    # STS
    # tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    model.eval().to(device)

    print('Evaluating...')
    results = se.eval(tasks)
    print(results)
    metrics = results
    # stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    # sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    # metrics = {"eval_stsb_spearman": stsb_spearman,
    #            "eval_sickr_spearman": sickr_spearman,
    #            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}

    # other
    # avg_transfer = 0
    # for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
    #     avg_transfer += results[task]['devacc']
    #     metrics['eval_{}'.format(task)] = results[task]['devacc']
    # avg_transfer /= 7
    # metrics['eval_avg_transfer'] = avg_transfer

    return metrics


ckpt_path = '../outputs/wiki_unsupervised_bert-base-uncased.ckpt'
model = SimCSETask.load_from_checkpoint(ckpt_path).model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
r = evaluate(model, tokenizer, 'cuda')
pprint(r)
