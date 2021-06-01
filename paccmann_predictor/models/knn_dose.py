import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
import logging

import time
from scipy.stats import pearsonr

def knn_dose(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    k: int = 1,
    return_knn_labels: bool = False,
    verbose: bool = False,
    result_path: str = None,
):
    """Baseline model for CPI prediction. Applies KNN classification using as
    similarity the Euclidean disance between cell representations (e.g. RNA-Seq)
    and FP similarity of the drugs.
    Predictions conceptually correspond to the predict_proba method of
    sklearn.neighbors.KNeighborsClassifier.

    Args:
        train_df (pd.DataFrame): DF with training samples in rows. Columns are:
            'drug', 'cell_line', 'dose', and 'label'.
        test_df (pd.DataFrame): DF with testing samples in rows. Columns are:
            'drug', 'cell_line', 'dose', and 'label'.
        drug_df (pd.DataFrame): DF with drug name as identifier and SMILES column.
        cell_df (pd.DataFrame): DF with cell line name as identifier and omic values
            as columns.
        k (int, optional): Hyperparameter for KNN classification. Defaults to 1.
        return_knn_labels (bool, optional): If set, the labels of the K nearest
            neighbors are also returned.
        verbose (bool, optional):
    """
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(drug_df, pd.DataFrame)
    assert isinstance(cell_df, pd.DataFrame)

    # Compute FPs of training data:
    drug_fp_dict = dict(
        zip(
            drug_df.index,
            [
                AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2)
                for smi in drug_df['SMILES'].values
            ],
        )
    )

    logger = logging.getLogger('knn_logger')

    predictions, knn_labels = [], []
    flipper = lambda x: x * -1 + 1

    start = time.time()
    # Compute all pairwise drug distances
    test_drugs = test_df.drug.unique()
    train_drugs = train_df.drug.unique()
    drug_dist_arr = np.zeros((len(train_drugs), len(test_drugs)))
    for i, test_drug in enumerate(test_drugs):
        fp = drug_fp_dict[test_drug]
        for j, train_drug in enumerate(train_drugs):
            if test_drug != train_drug: 
                drug_dist_arr[j,i]= flipper(
                            DataStructs.FingerprintSimilarity(fp, drug_fp_dict[train_drug])
                        )

    drug_dist_arr = pd.DataFrame(drug_dist_arr, columns = test_drugs, index = train_drugs)
    drug_dist_time = time.time()
    logger.info(f"drug_dist time = {drug_dist_time - start}")

    # Compute all pairwise cell line distances
    test_cell_lines = test_df.cell_line.unique()
    train_cell_lines = train_df.cell_line.unique()
    omics_dist_arr = np.zeros((len(train_cell_lines), len(test_cell_lines)))
    for i, test_cell_line in enumerate(test_cell_lines):
        for j, train_cell_line in enumerate(train_cell_lines):
            omics_dist_arr[j, i] = np.linalg.norm(
                        cell_df.loc[test_cell_line].values - cell_df.loc[train_cell_line].values
                    )

    omics_dist_arr = pd.DataFrame(omics_dist_arr, columns = test_cell_lines, index = train_cell_lines)
    omics_dist_time = time.time()
    logger.info(f"omics_dist time = {omics_dist_time - drug_dist_time}")

    train_df.set_index('cell_line', inplace=True)
    for drug_name, drug_rows in test_df.groupby('drug'):
        drug_start = time.time()
        drug_dist_df = drug_dist_arr[drug_name].to_frame()
        drug_dists = train_df.set_index('drug').join(drug_dist_df, how = 'left')[drug_name].values

        for cell_name, cell_rows in drug_rows.groupby('cell_line'):
            cell_start = time.time()
            cell_dist_df = omics_dist_arr[cell_name].to_frame()
            cell_dists = train_df.join(cell_dist_df, how = 'left')[cell_name].values
            # Normalize cell distances
            cell_dists = cell_dists / np.max(cell_dists)

            dose_dists = abs(cell_rows.dose.values[:, np.newaxis] - train_df.dose.values)
            # Normalize dose distances
            max_doses = np.amax(dose_dists, 1)[:, np.newaxis]
            dose_dists = (dose_dists / max_doses)

            dists = (dose_dists + drug_dists + cell_dists)

            if k == 1: 
                knns = np.argmin(dists, axis = 1)
                _knn_labels = np.array(train_df['label'])[knns]
            else: 
                knns = np.argsort(dists).transpose()[:k]
                _knn_labels = np.array(train_df['label'])[knns]
                _knn_labels = np.mean(_knn_labels, axis=0)
            
            predictions += _knn_labels.tolist()

            pearson = pearsonr(_knn_labels, cell_rows.label.values)
            logger.info(f'Pearson R = {pearson}')
          
            #cell_end = time.time()
            #logger.info(f'Time per cell line = {cell_end-cell_start}')
        drug_end = time.time()
        logger.info(f'Time per drug, all cell lines = {drug_end-drug_start}')

    return (predictions, knn_labels) if return_knn_labels else predictions
