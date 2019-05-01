# Pegasos SVM

Course Project for Optimization Methods (CS481) for Spring 2019 semester at IIIT Hyderabad.

This repository contains code for python implementation of binary / multiclass, Linear / Kernel Pegasos SVM classifier from this [paper](https://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)

### Prerequisites

Packages required to install the software

```
* Numpy
* tqdm
```

## Running code.

You have to download the dataset Fashion MNIST dataset from original repo [link](https://github.com/zalandoresearch/fashion-mnist)

The data must be stored in this folder inside a folder named 'fashionmnist'

Then you can run the code by executing below command from the home folder of this clone repo.
These codes are for linear SVM, kernel SVM and the linear SVM for multi class classification.
```
python Pegasos_linear.py
python kernel_pegasos.py
python Pegasos_multiclass.py
```

You may find commented code at the end of python file, try each of them to see the outputs and the change of parameters.

The link to some screen cast videos is: https://drive.google.com/drive/folders/1EDKXfjsm80JLrMUMEOeTZ5KVNdIfVd9Y?usp=sharing

* Built on Mac with Mojave and tested on Ubuntu 14.04

## Authors
* **Sangeeth Reddy** - *Linear, Kernalized Pegasos SVM*
* **Rudrabha Mukhopadhyay** - *Extension to MultiLabel, Report* 

## License

This project is only for educational / research purpose.
