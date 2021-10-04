# toy_gradlogp

This repo implements some toy examples of the following score matching algorithms in PyTorch:
* `ssm-vr`: [sliced score matching](https://arxiv.org/abs/1905.07088) with variance reduction
* `ssm`: [sliced score matching](https://arxiv.org/abs/1905.07088)
* `deen`: [deep energy estimator networks](https://arxiv.org/abs/1805.08306) (denoising score matching)

## Install
Basic requirements:
* Python >= 3.6
* TensorFlow >= 2.3.0
* PyTorch >= 1.8.0
* Numpy
* Matplotlib

<!-- Install from PyPI
```shell
pip install toy_gradlogp
``` -->

Or install the latest version from this repo
```shell
pip install git+https://github.com.Ending2015a/toy_gradlogp.git@master
```

## Examples

### Train an energy model

Type `--help` to see this message:
```
usage: train_energy.py [-h] [--logdir LOGDIR]
                       [--data {8gaussians,2spirals,checkerboard,rings}]
                       [--loss {ssm-vr,ssm,deen}]
                       [--noise {radermacher,sphere,gaussian}] [--lr LR]
                       [--size SIZE] [--eval_size EVAL_SIZE]
                       [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                       [--n_slices N_SLICES] [--gpu] [--log_freq LOG_FREQ]
                       [--eval_freq EVAL_FREQ] [--vis_freq VIS_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --logdir LOGDIR
  --data {8gaussians,2spirals,checkerboard,rings}
  --loss {ssm-vr,ssm,deen}
                        Loss type
  --noise {radermacher,sphere,gaussian}
                        Noise type
  --lr LR               learning rate
  --size SIZE           dataset size
  --eval_size EVAL_SIZE
                        dataset size for evaluation
  --batch_size BATCH_SIZE
                        training batch size
  --n_epochs N_EPOCHS   number of epochs to train
  --n_slices N_SLICES   number of slices for sliced score matching
  --gpu
  --log_freq LOG_FREQ
  --eval_freq EVAL_FREQ
  --vis_freq VIS_FREQ
```

Run `ssm-vr` on `2spirals` dataset
```shell
python -m examples.train_energy --gpu --loss ssm-vr --data 2spirals
```

## Results

### `ssm-vr`

### `deen`


