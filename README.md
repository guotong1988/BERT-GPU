# BERT MULTI GPU ON ONE MACHINE WITHOUT HOROVOD
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

# PRINCIPLE

More gpu means more data in a batch. And the gradients of a batch data is averaged for back-propagation.

If the learning rate of a batch is fixed, then the learning rate of a data is smaller.

If the learning rate of a data is fixed, then the learning rate of a batch is larger.

# REQUIREMENT

python 3

tensorflow 1.14 - 1.15

# TRAINING

0, edit the input and output file name in `create_pretraining_data.py` and `run_pretraining_gpu.py`

1, run `create_pretraining_data.py`

2, run `run_pretraining_gpu.py`

# PARAMETERS

Edit `n_gpus` in `run_pretraining_gpu.py`


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
