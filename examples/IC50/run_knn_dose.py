#!/usr/bin/env python3
"""Run KNN dose predictor."""
import argparse
import logging
import os
import pickle
import sys
from time import time

import numpy as np
from paccmann_predictor.models.knn_dose import knn_dose
from scipy.stats import pearsonr

import pandas as pd

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'train_sensitivity_filepath', type=str,
    help='Path to the drug sensitivity (IC50) data.'
)
parser.add_argument(
    'test_sensitivity_filepath', type=str,
    help='Path to the drug sensitivity (IC50) data.'
)
parser.add_argument(
    'gep_filepath', type=str,
    help='Path to the gene expression profile data.'
)
parser.add_argument(
    'smi_filepath', type=str,
    help='Path to the SMILES data.'
)
parser.add_argument(
    'gene_filepath', type=str,
    help='Path to a pickle object containing list of genes.'
)
parser.add_argument(
    'result_path', type=str,
    help='File where the results will be stored.'
)

def main(
    train_sensitivity_filepath, test_sensitivity_filepath, gep_filepath,
    smi_filepath, gene_filepath, result_path
):

	train_df = pd.read_csv(train_sensitivity_filepath).rename(columns = {'viability': 'label'})
	test_df = pd.read_csv(test_sensitivity_filepath).rename(columns = {'viability': 'label'})
	drug_df = pd.read_csv(smi_filepath, header=None, index_col = 1, names = ['SMILES'], sep = '\t')
	# Load the gene list
	with open(gene_filepath, 'rb') as f:
		gene_list = pickle.load(f)
	cell_df = pd.read_csv(gep_filepath, index_col = 0)
	shared_genes = list(set(gene_list) & set(cell_df.columns))
	cell_df = cell_df[shared_genes]

	predictions = knn_dose(train_df, test_df, drug_df, cell_df)

	pearson = pearsonr(predictions, test_df.label.values)
	print('Pearson R =', pearson)
	np.save(result_path, predictions)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the prediction
    main(
        args.train_sensitivity_filepath, args.test_sensitivity_filepath,
        args.gep_filepath, args.smi_filepath, args.gene_filepath,
        args.result_path
    )
