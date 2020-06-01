allennlp-beaker
---------------

An interactive CLI for submitting AllenNLP training jobs to beaker.

This handles all of the things you don't want to, like building a Docker image with the right environment (you'll be prompted to supply the exact version of AllenNLP that you need, as well as any other Python packages), and pushing it to beaker.

## Requirements

- [Beaker](https://github.com/beaker/docs/blob/master/docs/start/install.md#install)
- Docker
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
