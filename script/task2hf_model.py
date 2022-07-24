from task.simcse_task import SimCSETask

ckpt_path = 'C:\Projects\SemanticTextualSimilarity\outputs\sts_bert-base-cased-v2.ckpt'

task = SimCSETask.load_from_checkpoint(ckpt_path)
model = task.model
