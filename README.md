Rice
====

This is the repository of the coolest project ever! Clusterise, Categorise - Rice!

Prerequisites
-------------

- install `python >= 3.4.3`
- install `pip3 >= 1.5.6`
- run `$ pip3 install --upgrade -r requirements.txt`

Usage
-----

```
usage: main.py [-h] [-d DEVICE] [-t TRAINEPOCHS] [-b MINIBATCH] [-m MODE]
               [-f FLOATX] [-l LOG] [-o OUTPUT] [-r RESTORE] [-i ITER]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Train on this device
  -t TRAINEPOCHS, --trainepochs TRAINEPOCHS
                        Number of train epochs
  -b MINIBATCH, --minibatch MINIBATCH
                        Size of the minibatch
  -m MODE, --mode MODE  Theano run mode
  -f FLOATX, --floatX FLOATX
                        Theano floatX mode
  -l LOG, --log LOG     Log directory
  -o OUTPUT, --output OUTPUT
                        Trained model output directory
  -r RESTORE, --restore RESTORE
                        Path to the saved model to be continued from
  -i ITER, --iter ITER  Number of first epoch. Use together with --restore in
                        order to have beautiful logs

```

Creating reports
----------------

run `$ Rscript stat/report.R log/<NAME>.csv`. The `<NAME>.png` file will appear in the current folder.


Data
----
It is expected to have `data/` directory with the content of the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
archive as it is, preserving the original names of all files. The directory must not be included into git. 
