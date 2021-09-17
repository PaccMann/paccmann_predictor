import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from time import time
from scipy.stats import pearsonr


def knn(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    k: int = 1,
    return_knn_labels: bool = False,
    verbose: bool = False,
    result_path: str = None,
    chirality: bool = False,
    radius: int = 2,
):
    """Baseline model for CPI prediction. Applies KNN classification using as
    similarity the Euclidean disance between cell representations (e.g. RNA-Seq)
    and FP similarity of the drugs.
    Predictions conceptually correspond to the predict_proba method of
    sklearn.neighbors.KNeighborsClassifier.

    Args:
        train_df (pd.DataFrame): DF with training samples in rows. Columns are:
            'drug', 'cell_line' and 'label'.
        test_df (pd.DataFrame): DF with testing samples in rows. Columns are:
            'drug', 'cell_line' and 'label'.
        drug_df (pd.DataFrame): DF with drug name as identifier and SMILES column.
        cell_df (pd.DataFrame): DF with cell line name as identifier and omic values
            as columns.
        k (int, optional): Hyperparameter for KNN classification. Defaults to 1.
        return_knn_labels (bool, optional): If set, the labels of the K nearest
            neighbors are also returned.
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
                AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(smi), radius, useChirality=chirality
                )
                for smi in drug_df["SMILES"].values
            ],
        )
    )
    # Compute pairwise distances
    print(f"Computing pairwise distances of {len(cell_df)} expression profiles")
    cell_dist_dict = {}
    max_cell_dist = 0
    for cell_a_name, cell_a in tqdm(cell_df.T.items()):
        cell_dist_dict[cell_a_name] = {}
        for cell_b_name, cell_b in cell_df.T.items():
            d = np.linalg.norm(cell_a - cell_b)
            cell_dist_dict[cell_a_name][cell_b_name] = d
            if d > max_cell_dist:
                max_cell_dist = d

    # Will store computed distances to avoid re-computation
    tani_dict = {}

    predictions, drugs, cells, labels = [], [], [], []
    knn_labels_sample, knn_labels_full = [], []
    flipper = lambda x: x * -1 + 1
    t = time()
    train_labels = np.array(train_df["label"])
    for idx_loc, test_sample in tqdm(test_df.iterrows()):

        idx = test_df.index.get_loc(idx_loc)

        if verbose and idx % 10 == 0:
            print(f"Idx {idx}/{len(test_df)}")

        cell_name = test_sample["cell_line"]
        drug_name = test_sample["drug"]
        label = test_sample["label"]
        fp = drug_fp_dict[drug_name]

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

        get_cell_dist = lambda x: cell_dist_dict[cell_name][x]

        # new_cell = False
        # if cell_name not in omic_dict.keys():
        #     omic_dict[cell_name] = {}
        #     new_cell = True

        # if new_cell:

        #     def get_cell_dist(train_cell_name):

        #         if train_cell_name in omic_dict[cell_name].keys():
        #             return omic_dict[cell_name][train_cell_name]
        #         else:
        #             omic_dict[cell_name][train_cell_name] = np.linalg.norm(
        #                 cell_profile - cell_df_dict[train_cell_name]
        #             )
        #             return omic_dict[cell_name][train_cell_name]

        # else:

        #     def get_cell_dist(train_cell_name):
        #         return omic_dict[cell_name][train_cell_name]

        mol_dists, cell_dists = np.zeros((len(train_df),)), np.zeros((len(train_df),))

        # print(f"Rest took {time()-t}")
        # t = time()
        mol_dists = np.array(list(map(get_mol_dist, train_df["drug"].values)))
        # print(f" Mol dists took {time()-t}")
        # t = time()
        cell_dists = np.array(list(map(get_cell_dist, train_df["cell_line"].values)))
        # print(f"Cell dists took {time()-t}")
        # t = time()

        # Normalize cell distances
        cell_dists_sample = cell_dists / np.max(cell_dists)
        cell_dists_full = cell_dists / max_cell_dist

        knns_sample = np.argsort(mol_dists + cell_dists_sample)[:k]
        knns_full = np.argsort(mol_dists + cell_dists_full)[:k]

        _knn_labels_sample = train_labels[knns_sample]
        _knn_labels_full = train_labels[knns_full]

        knn_labels_sample.append(_knn_labels_sample)
        knn_labels_full.append(_knn_labels_full)

        predictions.append(
            (np.mean(_knn_labels_sample) + np.mean(_knn_labels_full)) / 2
        )

        drugs.append(drug_name)
        cells.append(cell_name)
        labels.append(label)

        if result_path is not None and idx % 100 == 0 and idx > 0:
            for x, y in zip(
                [knn_labels_sample, knn_labels_full], ["sample_norm", "full_norm"]
            ):
                df = pd.DataFrame(x)
                df.insert(0, "label", labels)
                df.insert(0, "cell", cells)
                df.insert(0, "drug", drugs)
                df.to_csv(os.path.join(result_path, f"knn_{y}_{idx}.csv"))
                p = pearsonr(labels, np.array(x).mean(axis=1))
                print(f"Running pearson ({y}): {round(p[0], 4)}")

    if result_path is not None:
        for x, y in zip(
            [knn_labels_sample, knn_labels_full], ["sample_norm", "full_norm"]
        ):
            df = pd.DataFrame(x)
            df.insert(0, "label", labels)
            df.insert(0, "cell", cells)
            df.insert(0, "drug", drugs)
            df.to_csv(os.path.join(result_path, f"knn_{y}_{idx}.csv"))
            p = pearsonr(labels, np.array(x).mean(axis=1))
            print(f"===Final pearson ({y}): {round(p[0], 4)}===")

    return (predictions, knn_labels_sample) if return_knn_labels else predictions
