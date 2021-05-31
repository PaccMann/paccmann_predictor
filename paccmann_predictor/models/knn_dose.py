import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

import time


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
    tani_dict_time = time.time()
    print("drug_dist time =", tani_dict_time - start)

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
    omic_dict_time = time.time()
    print("omics_dist time =", omic_dict_time - tani_dict_time)


    for drug_name, drug_rows in test_df.groupby('drug'):
        drug_start = time.time()
        drug_dist_df = drug_dist_arr[drug_name].to_frame()
        drug_dist_df['drug'] = drug_dist_df.index
        drug_dists = train_df.merge(drug_dist_df, on = 'drug', how = 'left')[drug_name].values
        for cell_name, cell_rows in drug_rows.groupby('cell_line'):
            cell_start = time.time()
            cell_dist_df = omics_dist_arr[cell_name].to_frame()
            cell_dist_df['cell_line'] = cell_dist_df.index
            cell_dists = train_df.merge(cell_dist_df, on = 'cell_line', how = 'left')[cell_name].values
            # Normalize cell distances
            cell_dists = cell_dists / np.max(cell_dists)
            for idx_loc, test_sample in tqdm(cell_rows.iterrows()):
                idx = test_df.index.get_loc(idx_loc)

                if verbose and idx % 10 == 0:
                    print(f'Idx {idx}/{len(test_df)}')

                dose_dists = np.abs(test_sample['dose'] - train_df['dose'].values)
                
                # Normalize dose distances
                dose_dists = dose_dists / np.max(dose_dists)

                knns = np.argsort(drug_dists + cell_dists + dose_dists)[:k]
                _knn_labels = np.array(train_df['label'])[knns]
                predictions.append(np.mean(_knn_labels))
                knn_labels.append(_knn_labels)
            cell_end = time.time()
            print('Time per cell line =', cell_end-cell_start)   
        drug_end = time.time()
        print('Time per drug =', drug_end-drug_start)

    return (predictions, knn_labels) if return_knn_labels else predictions
