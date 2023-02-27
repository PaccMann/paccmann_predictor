#!/usr/bin/env python3
"""Test PaccMann predictor."""
import argparse
import json
import logging
import os
import pickle
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from scipy.stats import pearsonr

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser()
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
    'model_filepath', type=str,
    help='Path to the stored model.'
)
parser.add_argument(
    'predictions_filepath', type=str,
    help='Path to the predictions.'
)
parser.add_argument(
    'params_filepath', type=str,
    help='Path to the parameter file.'
)
# yapf: enable


def main(
    test_sensitivity_filepath, gep_filepath,
    smi_filepath, gene_filepath, smiles_language_filepath, model_filepath, predictions_filepath,
    params_filepath
):

    logger = logging.getLogger('test')
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))


    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get('add_start_and_stop', True),
        padding=params.get('padding', True),
        padding_length=params.get('smiles_padding_length', None)
    )
    test_smiles_language = deepcopy(smiles_language)
    smiles_language.set_smiles_transforms(
        augment=params.get('augment_smiles', False),
        canonical=params.get('smiles_canonical', False),
        kekulize=params.get('smiles_kekulize', False),
        all_bonds_explicit=params.get('smiles_bonds_explicit', False),
        all_hs_explicit=params.get('smiles_all_hs_explicit', False),
        remove_bonddir=params.get('smiles_remove_bonddir', False),
        remove_chirality=params.get('smiles_remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('selfies', False)
    )
    test_smiles_language.set_smiles_transforms(
        augment=False,
        canonical=params.get('test_smiles_canonical', False),
        kekulize=params.get('smiles_kekulize', False),
        all_bonds_explicit=params.get('smiles_bonds_explicit', False),
        all_hs_explicit=params.get('smiles_all_hs_explicit', False),
        remove_bonddir=params.get('smiles_remove_bonddir', False),
        remove_chirality=params.get('smiles_remove_chirality', False),
        selfies=params.get('selfies', False),
        sanitize=params.get('selfies', False)
    )

    # Load the gene list
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)

    # Assemble test dataset
    test_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=test_smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        gene_expression_standardize=params.get(
            'gene_expression_standardize', True
        ),
        gene_expression_min_max=params.get('gene_expression_min_max', False),
        gene_expression_processing_parameters=params.get(
            'gene_expression_processing_parameters', {}
        ),
        device=torch.device(params.get('dataset_device', 'cpu')),
        iterate_dataset=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=params.get('num_workers', 0)
    )
    logger.info(
        f'Test dataset has {len(test_dataset)} samples with {len(test_loader)} batches'
    )

    device = get_device()
    logger.info(
        f'Device for data loader is {test_dataset.device} and for '
        f'model is {device}'
    )

    model_name = params.get('model_fn', 'paccmann')
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)
    try:
        logger.info(f'Attempting to restore model from {model_filepath}...')
        model.load(model_filepath, map_location=device)
    except Exception:
        raise ValueError(f'Error in restoring model from {model_filepath}!')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({'number_of_parameters': num_params})
    logger.info(f'Number of parameters {num_params}')

    # Start testing
    logger.info('Testing about to start... \n')
    model.eval()

    with torch.no_grad():
        test_loss = 0
        predictions = []
        # gene_attentions = []
        # epistemic_confs = []
        # aleatoric_confs = []
        labels = []
        for ind, (smiles, gep, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device), confidence = False
            )
            predictions.extend(list(y_hat.detach().cpu().squeeze().numpy()))
            # gene_attentions.append(pred_dict['gene_attention'])
            # epistemic_confs.append(pred_dict['epistemic_confidence'])
            # aleatoric_confs.append(pred_dict['aleatoric_confidence'])
            labels.extend(list(y.detach().cpu().squeeze().numpy()))
            loss = model.loss(y_hat, y.to(device))
            test_loss += loss.item()

    #gene_attentions = np.array([a.cpu().numpy() for atts in gene_attentions for a in atts])
    #epistemic_confs = np.array([c.cpu().numpy() for conf in epistemic_confs for c in conf]).ravel()
    #aleatoric_confs = np.array([c.cpu().numpy() for conf in aleatoric_confs for c in conf]).ravel()
    predictions = np.array(predictions)
    labels = np.array(labels)

    pearson = pearsonr(predictions, labels)[0]
    rmse = np.sqrt(np.mean((predictions - labels)**2))
    loss = test_loss / len(test_loader)
    logger.info(
        f"\t**RESULT**\t loss:{loss:.5f}, Pearson: {pearson:.3f}, RMSE: {rmse:.3f}"
    )

    df = test_dataset.drug_sensitivity_df
    df['prediction'] = predictions
    df.to_csv(predictions_filepath+'.csv')

    #np.save(predictions_filepath+'_gene_attention.npy', gene_attentions)
    #np.save(predictions_filepath+'_epistemic_confidence.npy', epistemic_confs)
    #np.save(predictions_filepath+'_aleatoric_confidence.npy', aleatoric_confs)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the testing
    main(
        args.test_sensitivity_filepath,
        args.gep_filepath, args.smi_filepath, args.gene_filepath,
        args.smiles_language_filepath, args.model_filepath, args.predictions_filepath, args.params_filepath
    )
