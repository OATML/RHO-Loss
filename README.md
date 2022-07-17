This is the code for the paper ["Prioritized training on points that are learnable, worth learning, and not yet learned"](https://arxiv.org/abs/2206.07137).

The code uses PyTorch Lightning, Hydra for config file management, and Weights & Biases for logging. The codebase is adapted from this [great template](https://github.com/ashleve/lightning-hydra-template).


# Installing dependencies
Conda: ```conda install --file my_environment.yaml```

Poetry: ```poetry install```

The repository also contains a singularity container definition file that can be built and used to run the experiments. See the ```singularity``` folder.

# Tutorial
```tutorial.ipynb``` contains the full training pipeline (irreducible loss model training and target model training) on CIFAR-10. This is the best place to start if you want to understand the code or reproduce our results.

# Codebase
The codebase contains the functionality for all the experiments in the paper (and more 😜).

## Irreducible loss model training
Start with ```run_irreducible.py```(which then calls ```src/train_irreducible.py```). The base config file is ```configs/irreducible_training.yaml```.

## Target model training
Start with ```run.py```(which then calls ```src/train.py```). The base config file is ```configs/config.yaml```. A key file is ```src//models/MultiModels.py```---this is the LightningModule that handles the training loop incl. batch selection. 

## More about the code
The datamodules are implemented in ```src/datamodules/datamodules.py```, the individual datasets in ```src/datamodules/dataset/sequence_datasets```. If you want to add your own dataset, note that ```__getitem__()``` needs to return the tuple ```(index, input, target)```, where ```index``` is the index of the datapoint with respect to the overall dataset (this is required so that we can match the irreducible losses to the correct datapoints).

All the selection methods mentioned in the paper (and more) are implemented in ```src/curricula/selection_methods.py```.

## ALBERT fine-tuning
All ALBERT experiments are implemented in a separate branch, which is a bit less clean. Good luck :-)

# Reproducibility
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

## Notes on using the importance sampling baseline:
To run the importance sampling experiments:

Importance sampling on CINIC10
``` 
python3 run_simple.py datamodule.data_dir=$DATA_DIR +experiment=importance_sampling_baseline.yaml 
```

## Let us know how it goes!
If you've tried RHO-LOSS and it worked well or not, or if you want us to give a presentation at your lab, we'd love to hear it! You can use the email at [www.soren-mindermann.com](https://www.soren-mindermann.com/).
