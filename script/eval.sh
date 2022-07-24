python -m script.evaluation \
  --ckpt_path outputs/wiki_unsupervised_bert-base-uncased-v3.ckpt \
  --tokenizer_name bert-base-uncased \
  --pooler cls \
  --task_set sts \
  --mode dev