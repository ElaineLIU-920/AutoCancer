# AutoCancer: An automated multi-modal framework for early cancer detection

AutoCancer is an automated multi-modal framework that leverages metaheuristic optimization and deep learning techniques for early cancer detection. Our framework is built upon the Transformer encoder and employs Bayesian optimization, seamlessly integrating feature selection, neural architecture search, and hyperparameter optimization in an automated and simultaneous manner. AutoCancer is designed to accept both one-dimensional (1D) and two-dimensional (2D) data as multi-modal inputs, accommodating each patient's variable and non-tabular features, such as single-nucleotide variants.

<img src='Overview of AutoCancer.svg' width=90%>

# Dataset
The preprocessed NSCLC datasets are in [dataset folder](./dataset/).

# Results 
The experiment results are in [result folder](./result/). 

# Analysis
The code for analysing with jupyter demos are in [analysis folder](./analysis/).

# Installation and Usage
[![python >3.8.13](https://img.shields.io/badge/python-3.8.13-brightgreen)](https://www.python.org/)
[![scipy-1.9.1](https://img.shields.io/badge/scipy-1.9.1-yellowgreen)](https://github.com/scipy/scipy) [![torch-1.12.1](https://img.shields.io/badge/torch-1.12.1-orange)](https://github.com/pytorch/pytorch) [![numpy-1.21.5](https://img.shields.io/badge/numpy-1.21.5-red)](https://github.com/numpy/numpy) [![pandas-1.5.1](https://img.shields.io/badge/pandas-1.5.1-lightgrey)](https://github.com/pandas-dev/pandas) [![scikit--optimize-0.9.0](https://img.shields.io/badge/scanpy-1.9.1-blue)](https://github.com/theislab/scanpy) [![scikit--learn-1.1.2](https://img.shields.io/badge/scikit--learn-1.1.2-green)](https://github.com/scikit-learn/scikit-learn)

The running environment of AutoCancer can be installed from Docker Hub registry:

1. Creat a copy this repository.
```bash
$ git clone https://github.com/ElaineLIU-920/AutoCancer.git
```

2. Download the docker image from Docker Hub.
```bash
$ docker pull linjingliu/autocancer:v0
```

3.  Start a container based on the image.
```bash
$ docker run --name autocancer --gpus all -it --rm -v <Replace with your local file path prefix>/AutoCancer:/AutoCancer linjingliu/autocancer:v0 /bin/bash
$ docker run --name autocancer1 --gpus all -it --rm -v /aaa/fionafyang/buddy1/elainelliu/AutoCancer:/AutoCancer linjingliu/autocancer:v0 /bin/bash
docker run --name autocancer --gpus all -it --rm -v /aaa/fionafyang/buddy1/elainelliu/AutoCancer:/AutoCancer mirrors.tencent.com/elainelliu/class:v1 /bin/bash

docker run --gpus all -it mirrors.tencent.com/elainelliu/class:v1 /bin/bash
```

4. Download datasets and checkpoint from provided links and replace the corresponding folder in scTranslator.
```bash
$ cd AutoCancer
```

5. Activate the enviroment.
```bash
$ conda activate pyenv
```

6. Demo for automated deep learning.
```bash
$ python ./code/AutoCancer.py \
--x_1d_train_path='./dataset/x_1d_train.pkl' \
--x_2d_train_path='./dataset/x_2d_train.pkl' \
--optimized_result_path='./result/optimized-result.pkl'
```

7. Demo for early cancer detection with optimized feature and network in Step 5.
```bash
# Inferrence with fine-tune
$ python ./code/early_cancer_detection.py \
--x_1d_train_path='./dataset/x_1d_train.pkl' \
--x_2d_train_path='./dataset/x_2d_train.pkl' \
--x_1d_test_path='./dataset/x_1d_test.pkl' \
--x_2d_test_path='./dataset/x_2d_test.pkl'
```

8. Demo for obtaining key gene mutation across all patients.
```bash
$ python ./code/gene_screening.py --num_top_gene=50 
```

9. Demo for obtaining combination gene mutation across all patients.
```bash
$ python ./code/gene_pair_discovery.py --num_top_gene_pair=50
```

# Time cost
The anticipated runtime for early cancer detection in a cohort of 256 NSCLC patients, utilizing an 8GB GPU, is approximately 60 seconds. This duration encompasses both the model training and inference processes.

# Disclaimer
This tool is for research purpose and not approved for clinical and commercial use.
All rights reserved.