#!/usr/bin/env python3
"""Train PaccMann predictor."""
import argparse
import json
import logging
import os
import pickle
import sys
from time import time
import numpy as np
# The logger imports tensorflow which may need to be imported before torch
# (importing tensorflow after torch causes problems dependent on the versions)
from paccmann_predictor.utils.logger import Logger
import torch
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.loss_functions import pearsonr
from paccmann_predictor.utils.utils import get_device

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
    help='Path to a pickle object a SMILES language object.'
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
    train_sensitivity_filepath, test_sensitivity_filepath, gep_filepath,
    smi_filepath, gene_filepath, smiles_language_filepath, model_path,
    params_filepath, training_name
):

    logger = logging.getLogger(f'{training_name}')
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory and dump files
    model_dir = os.path.join(model_path, training_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp, indent=4)

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Load the gene list
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)

    # Assemble datasets
    train_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=train_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        augment=params.get('augment_smiles', True),
        add_start_and_stop=params.get('smiles_start_stop_token', True),
        padding_length=params.get('smiles_padding_length', None),
        device=torch.device(params.get('dataset_device', 'cpu')),
        backend='eager'
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )

    test_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=test_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        augment=params.get('augment_smiles', True),
        add_start_and_stop=params.get('smiles_start_stop_token', True),
        padding_length=params.get('smiles_padding_length', None),
        device=torch.device(params.get('dataset_device', 'cpu')),
        backend='eager'
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )

    params.update(
        {
            'number_of_genes': len(gene_list),
            'smiles_vocabulary_size': smiles_language.number_of_tokens
        }
    )

    # Set the tensorboard logger
    tb_logger = Logger(os.path.join(model_dir, 'tb_logs'))
    device = get_device()
    logger.info(
        f'Device for data loader is {train_dataset.device} and for '
        f'model is {device}'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')

    model = MODEL_FACTORY[params.get('model_fn', 'mca')](params).to(device)

    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'Adam')]
        (model.parameters(), lr=params.get('lr', 0.01))
    )
    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp)

    # Start training
    logger.info('Training about to start...\n')
    t = time()
    min_loss, max_pearson = 100, 0

    for epoch in range(params['epochs']):

        model.train()
        logger.info(params_filepath.split('/')[-1])
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, gep, y) in enumerate(train_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device)
            )
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

        predictions = np.array(
            [p.cpu() for preds in predictions for p in preds]
        )
        labels = np.array([l.cpu() for label in labels for l in label])
        test_pearson_a = pearsonr(predictions, labels)
        test_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
        np.save(
            os.path.join(model_dir, 'results', 'preds.npy'),
            np.hstack([predictions, labels])
        )
        test_loss_a = test_loss / len(test_loader)
        logger.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss_a:.5f}, "
            f"Pearson: {test_pearson_a:.3f}, "
            f"RMSE: {test_rmse_a:.3f}"
        )

        # TensorBoard logging of scalars.
        info = {
            'train_loss': train_loss / len(train_loader),
            'test_loss': test_loss_a,
        }
        for tag, value in info.items():
            tb_logger.scalar_summary(tag, value, epoch + 1)

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, params.get('model_fn', 'mca')))
            if typ == 'best':
                logger.info(
                    f'\t New best performance in "{metric}"'
                    f' with value : {val:.7f} in epoch: {epoch}'
                )

        if test_loss_a < min_loss:
            min_loss = test_loss_a
            min_loss_pearson = test_pearson_a
            save(save_top_model, 'mse', 'best', min_loss)
            ep_loss = epoch
        if test_pearson_a > max_pearson:
            max_pearson = test_pearson_a
            max_pearson_loss = test_loss_a
            save(save_top_model, 'pearson', 'best', max_pearson)
            ep_pearson = epoch
        if (epoch + 1) % params.get('save_model', 100) == 0:
            save(save_top_model, 'epoch', str(epoch))
    logger.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
        f'\t (Pearson was {min_loss_pearson:4f}) \n \t'
        f'Pearson = {max_pearson:.4f} in epoch {ep_pearson} '
        f'\t (Loss was {max_pearson_loss:2f})'
    )
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.train_sensitivity_filepath, args.test_sensitivity_filepath,
        args.gep_filepath, args.smi_filepath, args.gene_filepath,
        args.smiles_language_filepath, args.model_path, args.params_filepath,
        args.training_name
    )
