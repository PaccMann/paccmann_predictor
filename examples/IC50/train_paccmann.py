#!/usr/bin/env python3
"""Train PaccMann predictor."""
import argparse
import json
import logging
import os
import pickle
import sys
from copy import deepcopy
from time import time

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.loss_functions import pearsonr
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer

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
    'smiles_language_filepath', type=str,
    help='Path to a folder with SMILES language .json files.'
)
parser.add_argument(
    'model_path', type=str,
    help='Directory where the model will be stored.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
parser.add_argument(
    'training_name', type=str,
    help='Name for the training.'
)
# yapf: enable


def main(
    train_sensitivity_filepath,
    test_sensitivity_filepath,
    gep_filepath,
    smi_filepath,
    gene_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
    training_name,
):

    logger = logging.getLogger(f"{training_name}")
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=params.get("smiles_padding_length", None),
    )
    test_smiles_language = deepcopy(smiles_language)
    smiles_language.set_smiles_transforms(
        augment=params.get("augment_smiles", False),
        canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get("test_smiles_canonical", True),
        kekulize=params.get("smiles_kekulize", False),
        all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False),
        selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )

    # Load the gene list
    with open(gene_filepath, "rb") as f:
        gene_list = pickle.load(f)

    # Assemble datasets
    train_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=train_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get("drug_sensitivity_min_max", True),
        drug_sensitivity_processing_parameters=params.get(
            "drug_sensitivity_processing_parameters", {}
        ),
        gene_expression_standardize=params.get("gene_expression_standardize", True),
        gene_expression_min_max=params.get("gene_expression_min_max", False),
        gene_expression_processing_parameters=params.get(
            "gene_expression_processing_parameters", {}
        ),
        device=torch.device(params.get("dataset_device", "cpu")),
        iterate_dataset=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )

    test_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get("drug_sensitivity_min_max", True),
        drug_sensitivity_processing_parameters=params.get(
            "drug_sensitivity_processing_parameters",
            train_dataset.drug_sensitivity_processing_parameters,
        ),
        gene_expression_standardize=params.get("gene_expression_standardize", True),
        gene_expression_min_max=params.get("gene_expression_min_max", False),
        gene_expression_processing_parameters=params.get(
            "gene_expression_processing_parameters",
            train_dataset.gene_expression_dataset.processing,
        ),
        device=torch.device(params.get("dataset_device", "cpu")),
        iterate_dataset=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params.get("num_workers", 0),
    )
    logger.info(
        f"Training dataset has {len(train_dataset)} samples, test set has "
        f"{len(test_dataset)}."
    )

    device = get_device()
    logger.info(
        f"Device for data loader is {train_dataset.device} and for "
        f"model is {device}"
    )
    save_top_model = os.path.join(model_dir, "weights/{}_{}_{}.pt")
    params.update(
        {  # yapf: disable
            "number_of_genes": len(gene_list),
            "smiles_vocabulary_size": smiles_language.number_of_tokens,
            "drug_sensitivity_processing_parameters": train_dataset.drug_sensitivity_processing_parameters,
            "gene_expression_processing_parameters": train_dataset.gene_expression_dataset.processing,
        }
    )
    model_name = params.get("model_fn", "paccmann_v2")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)

    if os.path.isfile(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt")):
        logger.info("Found existing model, restoring now...")
        model.load(os.path.join(model_dir, "weights", f"best_mse_{model_name}.pt"))

        with open(os.path.join(model_dir, "results", "mse.json"), "r") as f:
            info = json.load(f)

            min_rmse = info["best_rmse"]
            max_pearson = info["best_pearson"]
            min_loss = info["test_loss"]

    else:
        min_loss, min_rmse, max_pearson = 100, 1000, 0

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
        model.parameters(), lr=params.get("lr", 0.01)
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({"number_of_parameters": num_params})
    logger.info(f"Number of parameters {num_params}")
    logger.info(model)

    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, "model_params.json"), "w") as fp:
        json.dump(params, fp)

    # Start training
    logger.info("Training about to start...\n")
    t = time()

    model.save(save_top_model.format("epoch", "0", model_name))

    for epoch in range(params["epochs"]):

        model.train()
        logger.info(params_filepath.split("/")[-1])
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, gep, y) in enumerate(train_loader):
            y_hat, pred_dict = model(torch.squeeze(smiles.to(device)), gep.to(device))
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1e-6)
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        t = time()

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            test_loss = 0
            predictions = []
            labels = []
            for ind, (smiles, gep, y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    torch.squeeze(smiles.to(device)), gep.to(device)
                )
                predictions.append(y_hat)
                labels.append(y)
                loss = model.loss(y_hat, y.to(device))
                test_loss += loss.item()

        predictions = np.array([p.cpu() for preds in predictions for p in preds])
        labels = np.array([l.cpu() for label in labels for l in label])
        test_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
        test_rmse_a = np.sqrt(np.mean((predictions - labels) ** 2))
        test_loss_a = test_loss / len(test_loader)
        logger.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss_a:.5f}, "
            f"Pearson: {test_pearson_a:.3f}, "
            f"RMSE: {test_rmse_a:.3f}"
        )

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, model_name))
            with open(os.path.join(model_dir, "results", metric + ".json"), "w") as f:
                json.dump(info, f)
            np.save(
                os.path.join(model_dir, "results", metric + "_preds.npy"),
                np.vstack([predictions, labels]),
            )
            if typ == "best":
                logger.info(
                    f'\t New best performance in "{metric}"'
                    f" with value : {val:.7f} in epoch: {epoch}"
                )

        def update_info():
            return {
                "best_rmse": str(min_rmse),
                "best_pearson": str(float(max_pearson)),
                "test_loss": str(min_loss),
                "predictions": [float(p) for p in predictions],
            }

        if test_loss_a < min_loss:
            min_rmse = test_rmse_a
            min_loss = test_loss_a
            min_loss_pearson = test_pearson_a
            info = update_info()
            save(save_top_model, "mse", "best", min_loss)
            ep_loss = epoch
        if test_pearson_a > max_pearson:
            max_pearson = test_pearson_a
            max_pearson_loss = test_loss_a
            info = update_info()
            save(save_top_model, "pearson", "best", max_pearson)
            ep_pearson = epoch
        if (epoch + 1) % params.get("save_model", 100) == 0:
            save(save_top_model, "epoch", str(epoch))
    logger.info(
        "Overall best performances are: \n \t"
        f"Loss = {min_loss:.4f} in epoch {ep_loss} "
        f"\t (Pearson was {min_loss_pearson:4f}) \n \t"
        f"Pearson = {max_pearson:.4f} in epoch {ep_pearson} "
        f"\t (Loss was {max_pearson_loss:2f})"
    )
    save(save_top_model, "training", "done")
    logger.info("Done with training, models saved, shutting down.")


if __name__ == "__main__":
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.train_sensitivity_filepath,
        args.test_sensitivity_filepath,
        args.gep_filepath,
        args.smi_filepath,
        args.gene_filepath,
        args.smiles_language_filepath,
        args.model_path,
        args.params_filepath,
        args.training_name,
    )
