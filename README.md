# paccmann_predictor

PyTorch implementation of PaccMann.

IC50 prediction using drug properties and tissue-specific cell line (gene expression profiles).

`paccmann_predictor` is a package for drug sensitivity prediction and is the core component of the repo.

*NOTE*: PaccMann acronyms "Prediction of AntiCancer Compound sensitivity with Multi-modal Attention-based Neural Networks".

*NOTE*: This repo contains the `pytorch` implementation of our best model architecture (a multiscale convolutional attentive SMILES encoder).

*NOTE*: For details, please see our [paper](https://doi.org/10.1021/acs.molpharmaceut.9b00520) in *Molecular Pharmaceutics*.

## Requirements

- `conda>=3.7`

## Installation

The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 
To run the example training script we provide environment files under `examples/`.

Create a conda environment:

```sh
conda env create -f examples/conda.yml
```

Activate the environment:

```sh
conda activate paccmann_predictor
```

Install in editable mode for development:

```sh
pip install -e .
```

## Example usage

In the `examples` directory is a training script [train_paccmann.py](./examples/train_paccmann.py) that makes use
of `paccmann_predictor`.

```console
(paccmann_predictor) $ python examples/train_paccmann.py -h
usage: train_paccmann.py [-h]
                         train_sensitivity_filepath test_sensitivity_filepath
                         gep_filepath smi_filepath gene_filepath
                         smiles_language_filepath model_path params_filepath
                         training_name

positional arguments:
  train_sensitivity_filepath
                        Path to the drug sensitivity (IC50) data.
  test_sensitivity_filepath
                        Path to the drug sensitivity (IC50) data.
  gep_filepath          Path to the gene expression profile data.
  smi_filepath          Path to the SMILES data.
  gene_filepath         Path to a pickle object containing list of genes.
  smiles_language_filepath
                        Path to a pickle object a SMILES language object.
  model_path            Directory where the model will be stored.
  params_filepath       Path to the parameter file.
  training_name         Name for the training.

optional arguments:
  -h, --help            show this help message and exit
```

`params_path` could point to [examples/example_params.json](examples/example_params.json) examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_predictor` in your projects, please cite the following:

```bib
@article{doi:10.1021/acs.molpharmaceut.9b00520,
author = {Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian, Vigneshwari and Saez-Rodriguez, Julio and Rodriguez Martinez, Maria},
title = {Toward Explainable Anticancer Compound Sensitivity Prediction via Multimodal Attention-Based Convolutional Encoders},
journal = {Molecular Pharmaceutics},
year = {2019},
doi = {10.1021/acs.molpharmaceut.9b00520},
    note ={PMID: 31618586},
URL = { 
        https://doi.org/10.1021/acs.molpharmaceut.9b00520
},
eprint = { 
        https://doi.org/10.1021/acs.molpharmaceut.9b00520
}

}
@misc{born2019reinforcement,
    title={Reinforcement learning-driven de-novo design of anticancer compounds conditioned on biomolecular profiles},
    author={Jannis Born and Matteo Manica and Ali Oskooei and Maria Rodriguez Martinez},
    year={2019},
    eprint={1909.05114},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
