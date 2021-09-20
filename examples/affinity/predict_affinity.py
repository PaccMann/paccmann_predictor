"""Predict Affinity for a list of proteins and SMILES."""
import argparse
import json
import logging
import os
import sys

import pandas as pd
import torch
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.utils import get_device
from pytoda.files import read_smi
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.transforms import LeftPadding, ToTensor
from pytoda.datasets import SMILESTokenizerDataset
from paccmann_predictor.utils.interpret import (
    monte_carlo_dropout,
    test_time_augmentation,
)
from pytoda.smiles.transforms import AugmentTensor

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'model_path', type=str,
    help='Path to the trained model'
)
parser.add_argument(
    'protein_filepath', type=str,
    help='Path to a .smi file with protein sequences.'
)
parser.add_argument(
    'smi_filepath', type=str,
    help='Path to a .smi file with SMILES sequences.'
)
parser.add_argument(
    'output_folder', type=str,
    help='Directory where the output .csv will be stored.'
)
parser.add_argument(
    '-m', '--model_id', type=str,
    help='ID for model factory', default='bimodal_mca'
)
parser.add_argument(
    '-s', '--smiles_language_filepath', type=str, default='.',
    help='Path to a SMILES language object.'
)
parser.add_argument(
    '-p', '--protein_language_filepath', type=str, default='.',
    help='Path to a pickle of a Protein language object.'
)
parser.add_argument(
    '-l', '--label_filepath', type=str, default=None, required=False,
    help='Optional path to a file with labels'
)
parser.add_argument(
    '-c', '--confidence', action='store_true',
    help='Whether or not confidence predictions should be performed'
)
# yapf: enable


def main(
    model_path,
    protein_filepath,
    smi_filepath,
    output_folder,
    model_id,
    smiles_language_filepath,
    protein_language_filepath,
    label_filepath,
    confidence,
):

    logger = logging.getLogger('affinity_prediction')

    # Process parameter file:
    params = {}
    with open(os.path.join(model_path, 'model_params.json'), 'r') as fp:
        params.update(json.load(fp))

    # Create model directory
    os.makedirs(output_folder, exist_ok=True)

    device = get_device()
    weights_path = os.path.join(model_path, 'weights', 'best_ROC-AUC_bimodal_mca.pt')

    if label_filepath is not None:
        label_df = pd.read_csv(label_filepath, index_col=0)

    if smiles_language_filepath == '.':
        smiles_language_filepath = os.path.join(model_path, 'smiles_language.json')
    if protein_language_filepath == '.':
        protein_language_filepath = os.path.join(model_path, 'protein_language.pkl')
    # Load languages
    protein_language = ProteinLanguage.load(protein_language_filepath)
    smiles_language = SMILESTokenizer(
        vocab_file=smiles_language_filepath,
        padding=params.get('smiles_padding', True),
        padding_length=params.get('smiles_padding_length', None),
        add_start_and_stop=params.get('smiles_add_start_stop', True),
        augment=False,
        canonical=params.get('smiles_test_canonical', False),
        kekulize=params.get('smiles_kekulize', False),
        all_bonds_explicit=params.get('smiles_bonds_explicit', False),
        all_hs_explicit=params.get('smiles_all_hs_explicit', False),
        remove_bonddir=params.get('smiles_remove_bonddir', False),
        remove_chirality=params.get('smiles_remove_chirality', False),
        selfies=params.get('selfies', False),
    )
    augment = AugmentTensor(smiles_language)

    model = MODEL_FACTORY[model_id](params).to(device)

    if os.path.isfile(weights_path):
        try:
            model.load(weights_path, map_location=device)
        except Exception:
            raise ValueError(f'Error in model restoring from {weights_path}')
    else:
        logger.info(f'Did not find weights at {weights_path}, name weights "best.pt".')
    model.eval()

    # Transforms
    to_tensor = ToTensor()
    pad_seq = LeftPadding(model.protein_padding_length, protein_language.padding_index)

    # Read data
    sequences = read_smi(protein_filepath, names=['Sequence', 'Name'])
    ligands = read_smi(smi_filepath)

    smiles_data = SMILESTokenizerDataset(
        smi_filepath, smiles_language=smiles_language, iterate_dataset=False
    )
    smiles_loader = torch.utils.data.DataLoader(
        smiles_data, batch_size=256, drop_last=False, num_workers=0, shuffle=False
    )

    for idx, (sequence_id, row) in enumerate(sequences.iterrows()):
        logger.info(f'Target {idx+1}/{len(sequences)}: {sequence_id}')

        proteins = to_tensor(
            pad_seq(protein_language.sequence_to_token_indexes(row['Sequence']))
        ).unsqueeze(0)

        target_preds = []
        epi_confs, epi_preds, ale_confs, ale_preds = [], [], [], []
        for sidx, smiles_batch in enumerate(smiles_loader):
            protein_batch = proteins.repeat(len(smiles_batch), 1)
            preds, pred_dict = model(smiles_batch, protein_batch)
            target_preds.extend(preds.detach().squeeze().tolist())

            # Get confidences
            if confidence:

                ale_conf, ale_pred = test_time_augmentation(
                    model,
                    regime='tensors',
                    tensors=(smiles_batch, protein_batch),
                    augmenter=augment,
                    tensors_to_augment=0,
                )
                epi_conf, epi_pred = monte_carlo_dropout(
                    model, regime='tensors', tensors=(smiles_batch, protein_batch)
                )
                epi_confs.extend(epi_conf.detach().squeeze().tolist())
                epi_preds.extend(epi_pred.detach().squeeze().tolist())
                ale_confs.extend(ale_conf.detach().squeeze().tolist())
                ale_preds.extend(ale_pred.detach().squeeze().tolist())

        save_name = (
            sequence_id.strip()
            .replace(' ', '_')
            .replace('\\', '_')
            .replace('/', '_')
            .replace('=', '_')
        )
        df = pd.DataFrame({'SMILES': ligands['SMILES'], 'affinity': target_preds})
        if confidence:
            df['epistemic_confidence'] = epi_confs
            df['aleatoric_confidence'] = ale_confs
            df['epistemic_affinity'] = epi_preds
            df['aleatoric_affinity'] = ale_preds

        # Retrieve labels
        if label_filepath is not None:
            labels, ligand_names = [], []
            for smiles in ligands['SMILES']:
                try:
                    selected_row = label_df[
                        (
                            label_df['ligand_name']
                            == ligands[ligands['SMILES'] == smiles].index[0]
                        )
                        & (label_df['sequence_id'] == row['Name'])
                    ]
                    labels.append(selected_row['label'].values[0])
                    ligand_names.append(selected_row['ligand_name'].values[0])
                except IndexError:
                    labels.append(-1)
                    ligand_names.append(' ')
            df['ligand_name'] = ligand_names
            df['labels'] = labels

        df.to_csv(os.path.join(output_folder, f'{save_name}.csv'), index=False)

        # Free memory
        del preds, pred_dict

    logger.info('Done, shutting down.')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the predictions
    main(
        args.model_path,
        args.protein_filepath,
        args.smi_filepath,
        args.output_folder,
        args.model_id,
        args.smiles_language_filepath,
        args.protein_language_filepath,
        args.label_filepath,
        args.confidence,
    )
