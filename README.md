# BERT MULTI GPU
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

# MUST READ

Compare to https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT

,this repo is just a toy example. The NVIDIA repo should run on https://github.com/horovod/horovod 

And I will investigate to let this repo run without horovod with good speed.

# REQUIREMENT

python 3

tensorflow 1.14

# TRAINING

0, edit the input and output file name in `create_pretraining_data.py` and `run_pretraining_gpu.py`

1, run `create_pretraining_data.py`

2, run `run_pretraining_gpu.py`

# PARAMETERS

Edit `n_gpus` in `run_pretraining_gpu.py`

# DATA

In `sample_text.txt`, sentence is end by \n, paragraph is splitted by empty line.

# EXPERIMENT RESULT

Quora question pairs English dataset,

Official BERT: ACC 91.2, AUC 96.9

This BERT with pretrain loss 2.05: ACC 90.1, AUC 96.3
