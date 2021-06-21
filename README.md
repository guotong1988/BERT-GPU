# BERT MULTI-GPU PRE-TRAIN ON ONE MACHINE WITHOUT HOROVOD
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

# REASONABLE

More gpu means more data in a batch (, batch size is larger). And the gradients of a batch data is averaged for back-propagation.

If the sum learning rate of one batch is fixed, then the learning rate of one data is smaller, when batch size is larger.

If the learning rate of one data is fixed, then the sum learning rate of one batch is larger, when batch size is larger.

**Conclusion:** More gpu --> Larger sum learning rate of one batch --> Faster training.

# WHATS NEW

Using 1-GPU (100 batch size) vs using 4-GPU (400 batch size) for the same learning rate (0.00001) and same pre-training steps (1,000,000) will be no difference of 0.1% in downstream task.

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


