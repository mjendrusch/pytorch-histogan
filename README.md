# PyTorch-HistoGAN

Generative adversarial networks for histology images implemented using [PyTorch](https://github.com/pytorch/pytorch) + [torchsupport](https://github.com/mjendrusch/torchsupport).

## Installation

To install `histogan` for usage as a python package and to install all required dependencies, please run:

```
python setup.py install
```

in your python environment.

## Training

Run the `train.py` script on a directory containing training data, e.g.:

```
python train.py --prefix histogan-runs --name histogan --batch-size 32 msi-data
```
to train a GAN on image tiles from the training directory `msi-data` at batch-size 32, saving checkpoints in `histogan-runs/histogan`. The size of the generated images is determined by the size of images in the training dataset.

## Data Directory

The data directory should contain the following subdirectories:

```
data
|
----MSI
|
----MSS
```

containing tiles of H&E-stained slides of MSI tumours and MSS tumours respectively. The actual names of those subdirectories are irrelevant, just make sure that your data directory contains all MSI tiles in one folder and all MSS tiles in the other.

More generally, the `train.py` script can handle datasets of the form

```
data
|
----CLASS 1
|
----CLASS 2
|
...
|
----CLASS N
```


