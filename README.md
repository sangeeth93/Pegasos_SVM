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

And then edit the path of the parameter in get_data function with the absolute folder of the downloaded data.

Then you can run the code by executing below command from the home folder of this clone repo.

```
python kernel_pegasos.py
```

You may find commented code at the end of python file, try each of them to see the outputs and the change of parameters.


* Built on Mac with Mojave and tested on Ubuntu 14.04

## Authors
* **Sangeeth Reddy** - *Linear, Kernalized Pegasos SVM*
* **Rudrabha Mukhopadhyay** - *Extension to MultiLabel, Report* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
