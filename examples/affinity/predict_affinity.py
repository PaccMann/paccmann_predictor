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
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.transforms import LeftPadding, ToTensor
from pytoda.datasets import SMILESDataset

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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
    help='Path to a pickle of a SMILES language object.'
)
parser.add_argument(
    '-p', '--protein_language_filepath', type=str, default='.',
    help='Path to a pickle of a Protein language object.'
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
):

    logger = logging.getLogger('affinity_prediction')
    # Process parameter file:
    params = {}
    with open(os.path.join(model_path, 'model_params.json'), 'r') as fp:
        params.update(json.load(fp))

    # Create model directory
    os.makedirs(output_folder, exist_ok=True)

    device = get_device()
    weights_path = os.path.join(model_path, 'weights', 'best.pt')

    if smiles_language_filepath == '.':
        smiles_language_filepath = os.path.join(model_path, 'smiles_language.pkl')
    if protein_language_filepath == '.':
        protein_language_filepath = os.path.join(model_path, 'protein_language.pkl')
    # Load languages
    smiles_language = SMILESLanguage.load(smiles_language_filepath)
    protein_language = ProteinLanguage.load(protein_language_filepath)

    model = MODEL_FACTORY[model_id](params).to(device)

    if os.path.isfile(weights_path):
        try:
            model.load(weights_path, map_location=device)
        except Exception:
            logger.error(f'Error in model restoring from {weights_path}')
    else:
        logger.info(f'Did not find weights at {weights_path}, name weights "best.pt".')
    model.eval()

    # Transforms
    to_tensor = ToTensor()
    pad_seq = LeftPadding(model.protein_padding_length, protein_language.padding_index)

    # Read data
    sequences = read_smi(protein_filepath, names=['Sequence'])
    ligands = read_smi(smi_filepath)

    smiles_data = SMILESDataset(
        smi_filepath,
        smiles_language=smiles_language,
        padding_length=model.smiles_padding_length,
        sanitize=False,
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
        for smiles_batch in smiles_loader:
            protein_batch = proteins.repeat(len(smiles_batch), 1)
            preds, pred_dict = model(smiles_batch, protein_batch)
            target_preds.extend(preds.detach().squeeze().tolist())
        save_name = sequence_id.strip().replace(' ', '_').replace('\\', '_').replace('/', '_').replace('=','_')
        pd.DataFrame({'SMILES': ligands['SMILES'], 'affinity': target_preds}).to_csv(
            os.path.join(output_folder, f'{save_name}.csv'), index=False
        )

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
    )
