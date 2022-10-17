#!/usr/bin/env python3
"""Train Affinity predictor model."""
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from time import time

import numpy as np
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.datasets import DrugAffinityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer
from scipy.stats import pearsonr

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'train_affinity_filepath', type=str,
    help='Path to the drug affinity data.'
)
parser.add_argument(
    'test_affinity_filepath', type=str,
    help='Path to the drug affinity data.'
)
parser.add_argument(
    'protein_filepath', type=str,
    help='Path to the protein profile data.'
)
parser.add_argument(
    'smi_filepath', type=str,
    help='Path to the SMILES data.'
)
parser.add_argument(
    'smiles_language_filepath', type=str,
    help='Path to a json for a SMILES language object.'
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
    train_affinity_filepath,
    test_affinity_filepath,
    protein_filepath,
    smi_filepath,
    smiles_language_filepath,
    model_path,
    params_filepath,
    training_name,
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
    device = get_device()

    # Load languages
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    # Set transform
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
    smiles_language.set_encoding_transforms(
        padding=params.get('smiles_padding', True),
        padding_length=params.get('smiles_padding_length', None),
        add_start_and_stop=params.get('smiles_add_start_stop', True)
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
    test_smiles_language.set_encoding_transforms(
        padding=params.get('smiles_padding', True),
        padding_length=params.get('smiles_padding_length', None),
        add_start_and_stop=params.get('smiles_add_start_stop', True)
    )

    # Assemble datasets
    train_dataset = DrugAffinityDataset(
        drug_affinity_filepath=train_affinity_filepath,
        column_names=['ligand_name', 'sequence_id', 'affinity'],
        smi_filepath=smi_filepath,
        protein_filepath=protein_filepath,
        smiles_language=smiles_language,
        protein_amino_acid_dict=params.get('protein_amino_acid_dict', 'iupac'),
        protein_padding=params.get('protein_padding', True),
        protein_padding_length=params.get('protein_padding_length', None),
        protein_add_start_and_stop=params.get('protein_add_start_stop', True),
        protein_augment_by_revert=params.get('protein_augment', False),
        device=device,
        drug_affinity_dtype=torch.float,
        backend='eager',
        iterate_dataset=params.get('iterate_dataset', False)
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0),
    )

    test_dataset = DrugAffinityDataset(
        drug_affinity_filepath=test_affinity_filepath,
        column_names=['ligand_name', 'sequence_id', 'affinity'],
        smi_filepath=smi_filepath,
        protein_filepath=protein_filepath,
        smiles_language=smiles_language,
        smiles_padding=params.get('smiles_padding', True),
        smiles_padding_length=params.get('smiles_padding_length', None),
        smiles_add_start_and_stop=params.get('smiles_add_start_stop', True),
        protein_amino_acid_dict=params.get('protein_amino_acid_dict', 'iupac'),
        protein_padding=params.get('protein_padding', True),
        protein_padding_length=params.get('protein_padding_length', None),
        protein_add_start_and_stop=params.get('protein_add_start_stop', True),
        protein_augment_by_revert=False,
        device=device,
        drug_affinity_dtype=torch.float,
        backend='eager',
        iterate_dataset=params.get('iterate_dataset', False)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=params.get('num_workers', 0),
    )
    logger.info(
        f'Training dataset has {len(train_dataset)} samples, test set has '
        f'{len(test_dataset)}.'
    )

    logger.info(
        f'Device for data loader is {train_dataset.device} and for '
        f'model is {device}'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')
    protein_language = train_dataset.protein_sequence_dataset.protein_language
    params.update(
        {
            'smiles_vocabulary_size': smiles_language.number_of_tokens,
            'protein_vocabulary_size': protein_language.number_of_tokens,
        }
    )
    smiles_language.save_pretrained(model_dir)
    protein_language.save(os.path.join(model_dir, 'protein_language.pkl'))

    model_fn = params.get('model_fn', 'bimodal_mca')
    model = MODEL_FACTORY[model_fn](params).to(device)
    model._associate_language(smiles_language)
    model._associate_language(protein_language)

    if os.path.isfile(os.path.join(model_dir, 'weights', 'best_mca.pt')):
        logger.info('Found existing model, restoring now...')
        try:
            model.load(os.path.join(model_dir, 'weights', 'best_mca.pt'))

            with open(
                os.path.join(model_dir, 'results', 'mse.json'), 'r'
            ) as f:
                info = json.load(f)

                max_pearson = info['best_pearson']
                min_loss = info['test_loss']
                min_rmse = info['best_rmse']

        except Exception:
            min_loss, max_pearson, min_rmse = 10000, -1, 10000
    else:
        min_loss, max_pearson, min_rmse = 10000, -1, 10000

    # Define optimizer
    optimizer = OPTIMIZER_FACTORY[
        params.get('optimizer',
                   'adam')](model.parameters(), lr=params.get('lr', 0.001))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({'number_of_parameters': num_params})
    logger.info(f'Number of parameters: {num_params}')
    logger.info(f'Model: {model}')

    # Overwrite params.json file with updated parameters.
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        json.dump(params, fp)

    # Start training
    logger.info('Training about to start...\n')
    t = time()

    logger.info(train_dataset.smiles_dataset.smiles_language.transform_smiles)
    logger.info(
        train_dataset.smiles_dataset.smiles_language.transform_encoding
    )
    logger.info(test_dataset.smiles_dataset.smiles_language.transform_smiles)
    logger.info(test_dataset.smiles_dataset.smiles_language.transform_encoding)
    logger.info(train_dataset.protein_sequence_dataset.language_transforms)
    logger.info(test_dataset.protein_sequence_dataset.language_transforms)

    for epoch in range(params['epochs']):

        model.train()
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, proteins, y) in enumerate(train_loader):
            if ind % 1000 == 0:
                logger.info(f'Batch {ind}/{len(train_loader)}')
            y_hat, pred_dict = model(smiles, proteins)
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
            for ind, (smiles, proteins, y) in enumerate(test_loader):
                y_hat, pred_dict = model(
                    smiles.to(device), proteins.to(device)
                )
                predictions.append(y_hat)
                labels.append(y.clone())
                loss = model.loss(y_hat, y.to(device))
                test_loss += loss.item()

        predictions = torch.cat(predictions, dim=0).flatten().cpu().numpy()
        labels = torch.cat(labels, dim=0).flatten().cpu().numpy()

        test_loss = test_loss / len(test_loader)
        test_pearson = pearsonr(predictions, labels)[0]
        test_rmse = np.sqrt(np.mean((predictions - labels)**2))
        logger.info(
            f"\t **** TESTING **** Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {test_loss:.5f}, "
            f"Pearson: {test_pearson:.3f}, "
            f"RMSE: {test_rmse:.3f}"
        )

        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, model_fn))
            info = {
                'best_pearson': str(max_pearson),
                'best_rmse': str(min_rmse),
                'test_rmse': str(test_rmse),
                'test_pearson': str(test_pearson),
                'test_loss': str(min_loss),
            }
            with open(
                os.path.join(model_dir, 'results', metric + '.json'), 'w'
            ) as f:
                json.dump(info, f)
            np.save(
                os.path.join(model_dir, 'results', metric + '_preds.npy'),
                np.vstack([predictions, labels]),
            )
            if typ == 'best':
                logger.info(
                    f'\t New best performance in "{metric}"'
                    f' with value : {val:.7f} in epoch: {epoch}'
                )

        if test_pearson > max_pearson:
            max_pearson = test_pearson
            max_pearson_loss = test_loss
            save(save_top_model, 'pearson', 'best', max_pearson)
            ep_pearson = epoch
        if test_loss < min_loss:
            min_loss = test_loss
            min_rmse = test_rmse
            min_loss_pearson = test_pearson
            save(save_top_model, 'mse', 'best', min_loss)
            ep_loss = epoch
        if (epoch + 1) % params.get('save_model', 100) == 0:
            save(save_top_model, 'epoch', str(epoch))

    logger.info(
        'Overall best performances are: \n \t'
        f'Loss = {min_loss:.4f} in epoch {ep_loss} '
        f'\t (Pearson was {min_loss_pearson:4f}) \n \t'
        f'Pearson = {max_pearson:.4f} in epoch {ep_pearson} '
        f'\t (Loss was {max_pearson_loss:4f})'
    )
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.train_affinity_filepath, args.test_affinity_filepath,
        args.protein_filepath, args.smi_filepath,
        args.smiles_language_filepath, args.model_path, args.params_filepath,
        args.training_name
    )
