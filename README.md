# Overview

In this project, I have implemented the following research papers:

### From Incrementality in Deterministic Dependency Parsing (2004, Nivre)
- the arc-standard algorithm

### From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)

- feature extraction
- the neural network architecture including activation function
- loss function


# Installation

This project is implemented in python 3.6 and tensorflow 2.0. Follow these steps to setup your environment:

1. [Download and install Conda](http://https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-hw3 python=3.6
```

3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw3
```
4. Install the requirements:
```
pip install -r requirements.txt
```

5. Download glove wordvectors:
```
./download_glove.sh
```

# Data

The training, development and test set for dependency parsing is in conll format. The `train.conll` and `dev.conll` are labeled whereas `test.conll` is unlabeled.

For quick code development/debugging, there is a small fixture dataset. You can use this as training and development dataset while working on the code.

# Code Overview


## Train, Predict, Evaluate

There are three scripts in the repository `train.py`, `predict.py` and `evaluate.py` for training, predicting and evaluating a Dependency Parsing Model. You can supply `-h` flag to each of these to figure out how to use these scripts.

Here we show how to use the commands with the defaults.


#### Train a model
```
python train.py data/train.conll data/dev.conll

# stores the model by default at : serialization_dirs/default
```

#### Predict with model
```
python predict.py serialization_dirs/default \
                  data/dev.conll \
                  --predictions-file dev_predictions.conll
```

#### Evaluate model predictions

```
python evaluate.py serialization_dirs/default \
                   data/dev.conll \
                   dev_predictions.conll
```


## Dependency Parsing

  - `lib.model:` Defines the main model class of the neural dependency parser.

  - `lib.data.py`: Code dealing with reading, writing connl dataset, generating batches, extracting features and loading pretrained embedding file.

  - `lib.dependency_tree.py`: The dependency tree class file.

  - `lib.parsing_system.py`: This file contains the class for a transition-based parsing framework for dependency parsing.

  - `lib.configuration.py`: The configuration class file. Configuration reflects a state of the parser.

  - `lib.util.py`: This file contain function to load pretrained Dependency Parser.

  - `constants.py`: Sets project-wide constants for the project.


## Experimentations

I have tried different things for this task including changing the :

1. activations (cubic vs tanh vs sigmoid)
2. pretrained embeddings (GloVe embeddings vs using no embeddings)
3. tunability of embeddings (trainable embeddings vs frozen embeddings)

The findings are included in report.pdf.

The file `experiments.sh` enlists the commands you will need to train and save these models. In all you will need ~5 training runs, each taking about 30 minutes on cpu. See `colab_notes.md` to run experiments on gpu.

As shown in the `experiments.sh`, you can use `--experiment-name` argument in the `train.py` to store the models at different locations in `serialization_dirs`. You can also use `--cache-processed-data` and `--use-cached-data` flags in `train.py` to not generate the training features everytime. Please look at training script for details. Lastly, after training your dev results will be stored in serialization directory of the experiment with name `metric.txt`.

## References

1. [Joakim Nivre. 2004. Incrementality in deterministic dependency parsing. In Proceedings of the Workshop on Incremental Parsing: Bringing Engineering and Cognition Together (IncrementParsing ’04). Association for Computational Linguistics, USA, 50–57.
](https://www.aclweb.org/anthology/W04-0308.pdf)

2. [Danqi Chen et al. 2014. A Fast and Accurate Dependency Parser using Neural Networks](https://www.aclweb.org/anthology/D14-1082.pdf)