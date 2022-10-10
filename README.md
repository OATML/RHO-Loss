# Prioritized training on points that are learnable, worth learning, and not yet learned
Sören Mindermann*, Jan M Brauner*, Muhammed T Razzak*, Mrinank Sharma*, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, Yarin Gal 

| **[Abstract](#abstract)**
| **[Installation](#installation)**
  **[Tutorial](#tutorial)**
| **[Codebase](#codebase)**
| **[Citation](#citation)**

[![arXiv](https://img.shields.io/badge/arXiv-2106.02584-b31b1b.svg)](https://arxiv.org/abs/2206.07137)
[![Python 3.8](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.9-red.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

This is the code for the paper ["Prioritized training on points that are learnable, worth learning, and not yet learned"](https://proceedings.mlr.press/v162/mindermann22a.html).

The code uses PyTorch Lightning, Hydra for config file management, and Weights & Biases for logging. The codebase is adapted from this [great template](https://github.com/ashleve/lightning-hydra-template).

## Abstract
Training on web-scale data can take months. But much computation and time is wasted on redundant and noisy points that are already learnt or not learnable. To accelerate training, we introduce Reducible Holdout Loss Selection (RHO-LOSS), a simple but principled technique which selects approximately those points for training that most reduce the model's generalization loss. As a result, RHO-LOSS mitigates the weaknesses of existing data selection methods: techniques from the optimization literature typically select "hard" (e.g. high loss) points, but such points are often noisy (not learnable) or less task-relevant. Conversely, curriculum learning prioritizes "easy" points, but such points need not be trained on once learned. In contrast, RHO-LOSS selects points that are learnable, worth learning, and not yet learnt. RHO-LOSS trains in far fewer steps than prior art, improves accuracy, and speeds up training on a wide range of datasets, hyperparameters, and architectures (MLPs, CNNs, and BERT). On the large web-scraped image dataset Clothing-1M, RHO-LOSS trains in 18x fewer steps and reaches 2% higher final accuracy than uniform data shuffling.

## Installation
Conda: ```conda install --file my_environment.yaml```

Poetry: ```poetry install```

The repository also contains a singularity container definition file that can be built and used to run the experiments. See the ```singularity``` folder.

## Tutorial
```tutorial.ipynb``` contains the full training pipeline (irreducible loss model training and target model training) on CIFAR-10. This is the best place to start if you want to understand the code or reproduce our results.

## Codebase
The codebase contains the functionality for all the experiments in the paper (and more 😜).

### Irreducible loss model training
Start with ```run_irreducible.py```(which then calls ```src/train_irreducible.py```). The base config file is ```configs/irreducible_training.yaml```.

### Target model training
Start with ```run.py```(which then calls ```src/train.py```). The base config file is ```configs/config.yaml```. A key file is ```src//models/MultiModels.py```---this is the LightningModule that handles the training loop incl. batch selection. 

### More about the code
The datamodules are implemented in ```src/datamodules/datamodules.py```, the individual datasets in ```src/datamodules/dataset/sequence_datasets```. If you want to add your own dataset, note that ```__getitem__()``` needs to return the tuple ```(index, input, target)```, where ```index``` is the index of the datapoint with respect to the overall dataset (this is required so that we can match the irreducible losses to the correct datapoints).

All the selection methods mentioned in the paper (and more) are implemented in ```src/curricula/selection_methods.py```.

### ALBERT fine-tuning
All ALBERT experiments are implemented in a separate branch, which is a bit less clean. Good luck :-)

## Reproducibility
This repo can be used to reproduce all the experiments in the paper. Check out ```configs/experiment``` for some example experiment configs. The experiment files for the main results are: 
* CIFAR-10: ```cifar10_resnet18_irred.yaml``` and ```cifar10_resnet18_main.yaml```
* CINIC-10: ```cinic10_resnet18_irred.yaml``` and ```cinic10_resnet18_main.yaml```
* CIFAR-100: ```cifar100_resnet18_irred.yaml``` and ```cifar100_resnet18_main.yaml```
* Clothing-1M: ```c1m_resnet18_irred.yaml``` and ```c1m_resnet50_main.yaml```

NLP datasets, on a separate branch:
* CoLA:
  * Irreducible loss model training: ```python run_irreducible_nlp.py +experiment=nlp trainer.max_epochs=10 callbacks=val_loss datamodule.task_name=sst2 trainer.val_check_interval=0.05```
  * Target model training: ```python run_nlp.py +experiment=nlp datamodule.task_name=cola trainer.max_epochs=100 irreducible_loss_generator.f=\"path/to/file" selection_method_nlp=reducible_loss_selection```
* SST2:
  * Irreducible loss model training: ```python run_irreducible_nlp.py +experiment=nlp trainer.max_epochs=10 callbacks=val_loss datamodule.task_name=sst2 trainer.val_check_interval=0.05```
  * Target model training: ```python run_nlp.py +experiment=nlp trainer.max_epochs=15 datamodule.task_name=sst2 +trainer.val_check_interval=0.2 irreducible_loss_generator.f=\"path/to/file" selection_method_nlp=reducible_loss_selection ```

### Notes on using the importance sampling baseline:
To run the importance sampling experiments:

Importance sampling on CINIC10
``` 
python3 run_simple.py datamodule.data_dir=$DATA_DIR +experiment=importance_sampling_baseline.yaml 
```

## Citation

If you find this code helpful for your work, please cite our paper
[Paper](https://proceedings.mlr.press/v162/mindermann22a.html) as

```bibtex

@InProceedings{2022PrioritizedTraining,
  title = 	 {Prioritized Training on Points that are Learnable, Worth Learning, and not yet Learnt},
  author =       {Mindermann, S{\"o}ren and Brauner, Jan M and Razzak, Muhammed T and Sharma, Mrinank and Kirsch, Andreas and Xu, Winnie and H{\"o}ltgen, Benedikt and Gomez, Aidan N and Morisot, Adrien and Farquhar, Sebastian and Gal, Yarin},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {15630--15649},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/mindermann22a.html},}
```

## Let us know how it goes!
If you've tried RHO-LOSS and it worked well or not, or if you want us to give a presentation at your lab, we'd love to hear it! Correspondance to 'soren.mindermann at cs.ox.ac.uk'
