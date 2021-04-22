# BERT MULTI-GPU PRE-TRAIN ON ONE MACHINE WITHOUT HOROVOD
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

# PRINCIPLE

More gpu means more data in a batch (, batch size is larger). And the gradients of a batch data is averaged for back-propagation.

If the sum learning rate of one batch is fixed, then the learning rate of one data is smaller, when batch size is larger.

If the learning rate of one data is fixed, then the sum learning rate of one batch is larger, when batch size is larger.

**Conclusion:** more gpu --> larger sum learning rate of one batch.

# REQUIREMENT

python 3

tensorflow 1.14 - 1.15

# TRAINING

0, edit the input and output file name in `create_pretraining_data.py` and `run_pretraining_gpu.py`

1, run `create_pretraining_data.py`

2, run `run_pretraining_gpu_v2.py`

# PARAMETERS

Edit `n_gpus` in `run_pretraining_gpu_v2.py`


# DATA

In `sample_text.txt`, sentence is end by `\n`, paragraph is splitted by empty line.

# EXPERIMENT RESULT ON DOWNSTREAM TASKS

Quora question pairs English dataset,

Official BERT: ACC 91.2, AUC 96.9

This BERT with pretrain loss 2.05: ACC 90.1, AUC 96.3

# NOTE

### 1)
For `HierarchicalCopyAllReduce` `MirroredStrategy`, `global_step/sec` shows the sum of multi gpus' steps.
### 2)
`batch_size` is the `batch_size` per GPU, not the `global_batch_size`
