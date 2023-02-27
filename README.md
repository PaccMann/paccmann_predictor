[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml/badge.svg)](https://github.com/PaccMann/paccmann_predictor/actions/workflows/build.yml)

# paccmann_predictor

Drug interaction prediction with PaccMann.

`paccmann_predictor` is a package for drug interaction prediction, with examples of 
anticancer drug sensitivity prediction and drug target affinity prediction. Please see our papers:

- [_Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders_](https://doi.org/10.1021/acs.molpharmaceut.9b00520) (*Molecular Pharmaceutics*, 2019). This is the original paper on IC50 prediction using drug properties and tissue-specific cell line information (gene expression profiles). While the original code was written in `tensorflow` and is available [here](https://github.com/drugilsberg/paccmann), this is the `pytorch` implementation of the best PaccMann architecture (multiscale convolutional encoder).


**PaccMann for affinity prediction:**
- [Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2](https://iopscience.iop.org/article/10.1088/2632-2153/abe808) (_Machine Learning: Science and Technology_, 2021). In there, we propose a slightly modified version to predict drug-target binding affinities based on protein sequences and SMILES

![Graphical abstract](https://github.com/PaccMann/paccmann_predictor/blob/master/assets/paccmann.png "Graphical abstract")

## Installation
The library itself has few dependencies (see [setup.py](setup.py)) with loose requirements. 
First, set up the environment as follows:
```sh
conda env create -f examples/IC50/conda.yml
conda activate paccmann_predictor
pip install -e .
```


## Evaluate pretrained drug sensitivty model on your own data
First, please consider using our public [PaccMann webservice](https://ibm.biz/paccmann-aas) as described in the [NAR paper](https://academic.oup.com/nar/article/48/W1/W502/5836770).

To use our pretrained model, please download the model from: https://ibm.biz/paccmann-data (just download `models/single_pytorch_model`).
For example, assuming that you:
1. Set up your conda environment as described above;
2. Downloaded the model linked above in a directory called `single_pytorch_model` and
3. Downloaded the data from https://ibm.box.com/v/paccmann-pytoda-data in folders `data` and `splitted_data`;
then, the following command should work:
```console
(paccmann_predictor) $ python examples/IC50/test_paccmann.py \
splitted_data/gdsc_cell_line_ic50_test_fraction_0.1_id_997_seed_42.csv \
data/gene_expression/gdsc-rnaseq_gene-expression.csv \
data/smiles/gdsc.smi \
data/2128_genes.pkl \
single_pytorch_model/smiles_language \
single_pytorch_model/weights/best_mse_paccmann_v2.pt \
results \
single_pytorch_model/model_params.json
```
*NOTE*: If you bring your own data, please make sure to provide the omic data for the 2128 genes specified in `data/2128_genes.pkl`. Your omic data (here it is `data/gene_expression/gdsc-rnaseq_gene-expression.csv`) can contain more columns and it does not need to follow the order of the pickled gene list. But please dont change this pickle file. Also note that this is PaccMannV2 which is slightly improved compared to the paper version (context attention on both modalities).

## Finetuning on your own data
You can also **finetune** our pretrained model on your data instead of training a model from scratch. For that, please follow the instruction below for training on scratch and just set:
- `model_path` --> directory where the `single_pytorch_model` is stored
- `training_name` --> this should be `single_pytorch_model`
- `params_filepath` --> `base_path/single_pytorch_model/model_params.json`


## Training a model from scratch
To run the example training script we provide environment files under `examples/IC50/`.
In the `examples` directory is a training script [train_paccmann.py](./examples/IC50/train_paccmann.py) that makes use
of `paccmann_predictor`.

```console
(paccmann_predictor) $ python examples/IC50/train_paccmann.py -h
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

`params_filepath` could point to [examples/IC50/example_params.json](examples/IC50/example_params.json), examples for other files can be downloaded from [here](https://ibm.box.com/v/paccmann-pytoda-data).

## References

If you use `paccmann_predictor` in your projects, please cite the following:

```bib
@article{manica2019paccmann,
  title={Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders},
  author={Manica, Matteo and Oskooei, Ali and Born, Jannis and Subramanian, Vigneshwari and S{\'a}ez-Rodr{\'\i}guez, Julio and Mart{\'\i}nez, Mar{\'\i}a Rodr{\'\i}guez},
  journal={Molecular pharmaceutics},
  volume={16},
  number={12},
  pages={4797--4806},
  year={2019},
  publisher={ACS Publications},
  doi = {10.1021/acs.molpharmaceut.9b00520},
  note = {PMID: 31618586}
}

@article{born2021datadriven,
  author = {Born, Jannis and Manica, Matteo and Cadow, Joris and Markert, Greta and Mill, Nil Adell and Filipavicius, Modestas and Janakarajan, Nikita and Cardinale, Antonio and Laino, Teodoro and {Rodr{\'{i}}guez Mart{\'{i}}nez}, Mar{\'{i}}a},
  doi = {10.1088/2632-2153/abe808},
  issn = {2632-2153},
  journal = {Machine Learning: Science and Technology},
  number = {2},
  pages = {025024},
  title = {{Data-driven molecular design for discovery and synthesis of novel ligands: a case study on SARS-CoV-2}},
  url = {https://iopscience.iop.org/article/10.1088/2632-2153/abe808},
  volume = {2},
  year = {2021}
}
```
