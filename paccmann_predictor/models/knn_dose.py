import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def knn(
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
    # Will store computed distances to avoid re-computation
    tani_dict = {}
    omic_dict = {}

    predictions, knn_labels, drugs, cells = [], [], [], []
    flipper = lambda x: x * -1 + 1
    for idx_loc, test_sample in tqdm(test_df.iterrows()):

        idx = test_df.index.get_loc(idx_loc)

        if verbose and idx % 10 == 0:
            print(f'Idx {idx}/{len(test_df)}')

        cell_name = test_sample['cell_line']
        drug_name = test_sample['drug']
        fp = drug_fp_dict[drug_name]
        cell_profile = cell_df.loc[cell_name].values

        new_mol = False
        if drug_name not in tani_dict.keys():
            tani_dict[drug_name] = {}
            new_mol = True

        if new_mol:

            def get_mol_dist(train_drug):
                if train_drug in tani_dict[drug_name].keys():
                    return tani_dict[drug_name][train_drug]
                else:
                    tani_dict[drug_name][train_drug] = flipper(
                        DataStructs.FingerprintSimilarity(fp, drug_fp_dict[train_drug])
                    )
                    return tani_dict[drug_name][train_drug]

        else:

            def get_mol_dist(train_drug):
                return tani_dict[drug_name][train_drug]

        new_cell = False
        if cell_name not in omic_dict.keys():
            omic_dict[cell_name] = {}
            new_cell = True

        if new_cell:

            def get_cell_dist(train_cell_name):

                if train_cell_name in omic_dict[cell_name].keys():
                    return omic_dict[cell_name][train_cell_name]
                else:
                    omic_dict[cell_name][train_cell_name] = np.linalg.norm(
                        cell_profile - cell_df.loc[train_cell_name].values
                    )
                    return omic_dict[cell_name][train_cell_name]

        else:

            def get_cell_dist(train_cell_name):
                return omic_dict[cell_name][train_cell_name]

        mol_dists, cell_dists = np.zeros((len(train_df),)), np.zeros((len(train_df),))

        mol_dists = np.array(list(map(get_mol_dist, train_df['drug'].values)))
        cell_dists = np.array(list(map(get_cell_dist, train_df['cell_line'].values)))
        dose_dists = np.abs(test_df['dose'] - train_df['dose'].values)

        # Normalize cell distances
        cell_dists = cell_dists / np.max(cell_dists)
        # Normalize dose distances
        dose_dists = dose_dists / np.max(dose_dists)

        knns = np.argsort(mol_dists + cell_dists + dose_dists)[:k]
        _knn_labels = np.array(train_df['label'])[knns]
        predictions.append(np.mean(_knn_labels))
        knn_labels.append(_knn_labels)
        drugs.append(drug_name)
        cells.append(cell_name)

        if result_path is not None and idx % 10 == 0:
            df = pd.DataFrame(knn_labels)
            df.insert(0, 'cell', cells)
            df.insert(0, 'drug', drugs)
            df.to_csv(os.path.join(result_path, f'knn_{idx}.csv'))

    if result_path is not None:
        df = pd.DataFrame(knn_labels)
        df.insert(0, 'cell', cells)
        df.insert(0, 'drug', drugs)
        df.to_csv(os.path.join(result_path, f'knn_{idx}.csv'))

    return (predictions, knn_labels) if return_knn_labels else predictions
