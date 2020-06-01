allennlp-beaker
----

An interactive AllenNLP plugin for submitting training jobs to beaker

## Requirements

- [Beaker](https://github.com/beaker/docs/blob/master/docs/start/install.md#install)
- Python 3.6 or 3.7

## Installation

```bash
pip install git+https://github.com/allenai/allennlp-beaker.git
```

## Usage

Submit a training job to beaker with

```bash
allennlp-beaker PATH_TO_TRAINING_CONFIG.jsonnet
```

and then follow the prompts. For more information, run the help command:

```bash
allennlp-beaker --help
```
