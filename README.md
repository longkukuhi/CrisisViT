# CrisisViT

CrisisViT adapts a vision transformer to classify social-media images during disasters. The paper evaluates four downstream tasks: disaster type, informativeness, humanitarian category and damage severity. This repository includes PyTorch training scripts and four encoder checkpoints pretrained on Incidents1M. [Read the paper](https://arxiv.org/abs/2401.02838).

## Choose a pretrained checkpoint

| Pretraining signal on Incidents1M | Weights |
| --- | --- |
| Incident and place labels jointly | [Download checkpoint](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EcZ0GDd28xFAkpSHNgal588BGM_aj_U_ujF_FxCCHufv5A?e=B5Wql1) |
| Incident labels | [Download checkpoint](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EX3LWhOpmLFNniLH08dzI6sBpJJs3s2Wp0XYQ1r4VWPNNg?e=bVCRoT) |
| Place labels | [Download checkpoint](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/ETrtXvO_dWpHpLb3-VuC1-0B7ZNrs1uikQq-2jSyjgrkrw?e=XRYSOq) |
| Unsupervised pretraining | [Download checkpoint](https://gla-my.sharepoint.com/:u:/g/personal/z_long_2_research_gla_ac_uk/EZln7ue-u2ZHsyYrc8uVzNABB5xXECYJkdZXOhgM-q3gaQ?e=QeIzjs) |

## Fine-tune on a crisis classification task

1. Prepare the images and the train, development and test TSV files for all four downstream tasks. Set their paths in [configs/downstream.yaml](configs/downstream.yaml). The configuration groups the split files by task.
2. Set up a CUDA PyTorch environment with the dependencies imported by the scripts, including torchvision, timm, ruamel.yaml and wandb. The repository supplies experiment configurations and does not pin a complete Python environment.
3. Set the downloaded encoder checkpoint path and launch from the repository root. The downstream script uses a 224-pixel input by default.

```bash
python finetune.py --config configs/downstream.yaml \
  --checkpoint /path/to/pretrained-encoder.pth \
  --output_dir output/downstream
```

The script records its configuration under the output directory and saves the best validation checkpoint during training. To inspect the model and loading logic, see [modeling/crisis_vit.py](modeling/crisis_vit.py) and [finetune.py](finetune.py).

## Pretraining and release history

The repository also includes [MAE pretraining](main_pretrain.py) with [its configuration](configs/mae.yaml), plus [crisis pretraining](crisis_pretrain.py) and [binary pretraining](binary_pretrain.py). The binary pretraining script reads Incidents1M split paths from [configs/pretrain.yaml](configs/pretrain.yaml). Code was released in March 2023 and pretrained weights in August 2024.

**Reference** — Zijun Long, Richard McCreadie and Muhammad Imran. *CrisisViT: A Robust Vision Transformer for Crisis Image Classification*. [Paper](https://arxiv.org/abs/2401.02838).
