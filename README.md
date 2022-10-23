# UMeI

U-shaped network for Medical Imaging

## Setup Environment

```zsh
conda env create -n umei
conda activate umei
echo "PYTHONPATH=`pwd`" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# we use wandb as default logger
wandb login
```
## Run Pre-training

```bash
python scripts/snim/main.py conf/snim/ct/<config file name> --mask_block_shape <mask block shape> --visible_factor <λ> --datasets amos
```

## Fine-tuning on BTCV

```bash
python scripts/btcv/main.py conf/btcv/<config file name> --do_train --pretrain_path <check point path>
```
