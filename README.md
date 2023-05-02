# Reliable Gradient-free and Likelihood-free Prompt Tuning
This repository contains official implementation of EACL 2023 (Findings) paper [Reliable Gradient-free and Likelihood-free Prompt Tuning](https://aclanthology.org/2023.findings-eacl.183/).

## Prepare your environment

```bash
conda create --name SBI_LLM python=3.8
conda activate SBI_LLM
pip install transformers==4.1.1
pip install datasets
pip install fastNLP
pip install cma
pip install sklearn
pip install sbi
pip install uq360
```

## Usage
The main.py file includes all the interfaces required for experiments.

The algorithm.py file implements 4 algorithms proposed in the work: 
1. **Ensembles**: prompt ensembles based on CMA_ES algorithm (for *gradient-free* prompt tuning).
2. **CMA_ELBO**: gradient-free variational inference based on CMA_ES algorithm (for *gradient-free* prompt tuning).
3. **ABC_SMC**: SBI based algorithm ABC_SMC (for *likelihood-free* prompt tuning).
4. **SBI_neural**: neural net-based SBI algorithm (for *likelihood-free* prompt tuning).
### Examples:
To apply ABC_SMC algorithm for *likelihood-free* prompt tuning on SST2 dataset, and collect 100 prompt samples, simply run:
```bash
python main.py \
  --task_name "sst2" \
  --alg_name "ABC_SMC"\
  --num_samples 100 \
  --device "cuda:0" \
  --seed 0 \
```
To apply Prompt Ensembles algorithm for *gradient-free* prompt tuning on SNLI dataset, and collect 10 prompt samples, simply run:
```bash
python main.py \
  --task_name "snli" \
  --alg_name "Ensembles"\
  --num_samples 10 \
  --device "cuda:0" \
  --seed 0 \
```

## Reference

The implementation is based on this repo: https://github.com/txsun1997/Black-Box-Tuning.

## Citations
```bash
@inproceedings{shen2023reliable,
  title={Reliable Gradient-free and Likelihood-free Prompt Tuning},
  author={Shen, Maohao and Ghosh, Soumya Sankar and Sattigeri, Prasanna and Das, Subhro and Bu, Yuheng and Wornell, Gregory},
  booktitle={Findings of the Association for Computational Linguistics: EACL 2023},
  pages={2371--2384},
  year={2023}
}
```

