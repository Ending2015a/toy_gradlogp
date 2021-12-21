
<img src="https://user-images.githubusercontent.com/18180004/136144615-0cd92028-8226-40c1-81ee-fa6c067e91e3.png" align="right" width="25%"/>

# toy_gradlogp

This repo implements some toy examples of the following score matching algorithms in PyTorch:
* `ssm-vr`: [sliced score matching](https://arxiv.org/abs/1905.07088) with variance reduction
* `ssm`: [sliced score matching](https://arxiv.org/abs/1905.07088)
* `deen`: [deep energy estimator networks](https://arxiv.org/abs/1805.08306)
* `dsm`: [denoisnig score matching](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)


Related projects:
* [toy_gradlogp_tf2](https://github.com/Ending2015a/toy_gradlogp_tf2): TensorFlow 2.0 Implementation.

## Installation
Basic requirements:
* Python >= 3.6
* TensorFlow >= 2.3.0
* PyTorch >= 1.8.0

Install from PyPI
```shell
pip install toy_gradlogp
```

Or install the latest version from this repo
```shell
pip install git+https://github.com.Ending2015a/toy_gradlogp.git@master
```

## Examples
The examples are placed in [toy_gradlogp/run/](https://github.com/Ending2015a/toy_gradlogp/tree/master/toy_gradlogp/run)

### Train an energy model

Run `ssm-vr` on `2spirals` dataset (don't forget to add `--gpu` to enable gpu)
```shell
python -m toy_gradlogp.run.train_energy --gpu --loss ssm-vr --data 2spirals
```

To see the full options, type `--help` command:
```
python -m toy_gradlogp.run.train_energy --help
```

```
usage: train_energy.py [-h] [--logdir LOGDIR]
                       [--data {8gaussians,2spirals,checkerboard,rings}]
                       [--loss {ssm-vr,ssm,deen,dsm}]
                       [--noise {radermacher,sphere,gaussian}] [--lr LR]
                       [--size SIZE] [--eval_size EVAL_SIZE]
                       [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                       [--n_slices N_SLICES] [--n_steps N_STEPS] [--eps EPS]
                       [--gpu] [--log_freq LOG_FREQ] [--eval_freq EVAL_FREQ]
                       [--vis_freq VIS_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --logdir LOGDIR
  --data {8gaussians,2spirals,checkerboard,rings}
                        dataset
  --loss {ssm-vr,ssm,deen,dsm}
                        loss type
  --noise {radermacher,sphere,gaussian}
                        noise type
  --lr LR               learning rate
  --size SIZE           dataset size
  --eval_size EVAL_SIZE
                        dataset size for evaluation
  --batch_size BATCH_SIZE
                        training batch size
  --n_epochs N_EPOCHS   number of epochs to train
  --n_slices N_SLICES   number of slices for sliced score matching
  --n_steps N_STEPS     number of steps for langevin dynamics
  --eps EPS             noise scale for langevin dynamics
  --gpu                 enable gpu
  --log_freq LOG_FREQ   logging frequency (unit: epoch)
  --eval_freq EVAL_FREQ
                        evaluation frequency (unit: epoch)
  --vis_freq VIS_FREQ   visualization frequency (unit: epoch)
```

## Results

Tips: The larger density has a lower energy!

### `8gaussians`

| Algorithm | Results|
|-|-|
|`ssm-vr`|![](/assets/ssm-vr_8gaussians.png)|
|`ssm`|![](/assets/ssm_8gaussians.png)|
|`deen`| ![](/assets/deen_8gaussians.png) |
|`dsm`| ![](/assets/dsm_8gaussians.png) |

### `2spirals`

| Algorithm | Results|
|-|-|
|`ssm-vr`|![](/assets/ssm-vr_2spirals.png)|
|`ssm`|![](/assets/ssm_2spirals.png)|
|`deen`| ![](/assets/deen_2spirals.png) |
|`dsm`| ![](/assets/dsm_2spirals.png) |

### `checkerboard`
| Algorithm | Results|
|-|-|
|`ssm-vr`|![](/assets/ssm-vr_checkerboard.png)|
|`ssm`|![](/assets/ssm_checkerboard.png)|
|`deen`| ![](/assets/deen_checkerboard.png) |
|`dsm`| ![](/assets/dsm_checkerboard.png) |

### `rings`
| Algorithm | Results|
|-|-|
|`ssm-vr`|![](/assets/ssm-vr_rings.png)|
|`ssm`|![](/assets/ssm_rings.png)|
|`deen`| ![](/assets/deen_rings.png) |
|`dsm`| ![](/assets/dsm_rings.png) |
