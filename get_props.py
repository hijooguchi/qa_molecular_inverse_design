import tensorflow as tf
from os.path import join, isdir
from os import makedirs, environ
import random
import argparse
from glob import glob
import numpy as np
import pickle
import json
import pandas as pd
from typing import List, Tuple, Dict, Union
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import (
    MolWt,
    MolLogP,
    TPSA,
    NumRotatableBonds,
    NumAromaticRings,
)
from rdkit.Chem.rdMolDescriptors import CalcNumLipinskiHBD, CalcNumLipinskiHBA
from tqdm import tqdm

from external_lib.sas import sascorer
from external_lib.cddd.inference import InferenceModel


RDLogger.DisableLog('rdApp.*')


def fix_seed(seed: int) -> None:
    environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def get_physical_properties(
    smiles_list: Union[Tuple[str], List[str]],
) -> Dict[str, np.ndarray]:
    """
    Calculate following molecular physical properties:
        - QED (Quantitative Estimate of Drug-likeness)
        - MW (Molecular Weight)
        - logP (Octanol-water Partition Coefficient)
        - TPSA (Topological Polar Surface Area)
        - nHBD (Number of Hydrogen Bond Donors)
        - nHBA (Number of Hydrogen Bond Acceptors)
        - nRB (Number of Rotatable Bonds)
        - nAB (Number of Aromatic Bonds)
        - SAS (Synthetic Accessibility Score)
    """
    physical_props = {}
    qed_list = []
    mw_list = []
    logp_list = []
    tpsa_list = []
    nhbd_list = []
    nhba_list = []
    nrb_list = []
    nar_list = []
    sas_list = []

    for smiles in tqdm(smiles_list, total=len(smiles_list)):
        if smiles == '':
            mol = None
        else:
            mol = MolFromSmiles(smiles)

        if mol is None:
            qed_list.append(np.nan)
            mw_list.append(np.nan)
            logp_list.append(np.nan)
            tpsa_list.append(np.nan)
            nhbd_list.append(np.nan)
            nhba_list.append(np.nan)
            nrb_list.append(np.nan)
            nar_list.append(np.nan)
            sas_list.append(np.nan)
        else:
            qed_list.append(qed(mol))
            mw_list.append(MolWt(mol))
            logp_list.append(MolLogP(mol))
            tpsa_list.append(TPSA(mol))
            nhbd_list.append(CalcNumLipinskiHBD(mol))
            nhba_list.append(CalcNumLipinskiHBA(mol))
            nrb_list.append(NumRotatableBonds(mol))
            nar_list.append(NumAromaticRings(mol))
            sas_list.append(sascorer.calculateScore(mol))

    qed_list = np.array(qed_list, dtype=np.float32)
    valid_index = np.where(~np.isnan(qed_list))[0].astype(np.int32)

    physical_props['QED'] = qed_list[valid_index]
    physical_props['MW'] = np.array(mw_list, dtype=np.float32)[valid_index]
    physical_props['logP'] = np.array(logp_list, dtype=np.float32)[valid_index]
    physical_props['TPSA'] = np.array(tpsa_list, dtype=np.float32)[valid_index]
    physical_props['nHBD'] = np.array(nhbd_list, dtype=np.float32)[valid_index]
    physical_props['nHBA'] = np.array(nhba_list, dtype=np.float32)[valid_index]
    physical_props['nRB'] = np.array(nrb_list, dtype=np.float32)[valid_index]
    physical_props['nAR'] = np.array(nar_list, dtype=np.float32)[valid_index]
    physical_props['SAS'] = np.array(sas_list, dtype=np.float32)[valid_index]
    physical_props['valid_index'] = valid_index

    return physical_props


def get_use_size_of_each_file(
    dataset_dir: str,
    dataset_size: int,
) -> pd.DataFrame:
    df_count = pd.read_csv(
        join('external_lib', 'dataset_detailed_count.csv'),
        index_col=0,
    )
    drop_data_list = [
        "M700",
        "M800",
        "M900",
        "P500",
        "P600",
        "P700",
        "P800",
        "P900",
    ]
    df_count = df_count[~df_count["LogP"].isin(drop_data_list)]

    data_volume = df_count['n_SMILES'].sum()
    df_count.loc[:, 'use_data_size'] = (
            dataset_size * df_count['n_SMILES'] / data_volume
    ).round().astype(int)

    volume_diff = dataset_size - df_count['use_data_size'].sum()
    if volume_diff != 0:
        df_count.loc[
            df_count['use_data_size'] == df_count['use_data_size'].max(),
            'use_data_size'
        ] += volume_diff

    df_count['file_name'] = (
        df_count['n_HeavyAtomCount'] + df_count['LogP'] + '.pickle'
    )

    return df_count.set_index('file_name')[['use_data_size']]


def get_dataset(
    dataset_dir: str,
    train_size: int,
    test_size: int,
    is_load_all: bool = False,
) -> Tuple[Tuple[str], Tuple[str]]:
    # logP between -7 and 5
    dataset_paths = (
        glob(
            join(dataset_dir, 'H??', 'H??P[0-4]??.pickle')
        ) +
        glob(
            join(dataset_dir, 'H??', 'H??M[0-6]??.pickle')
        )
    )
    dataset_paths.sort()

    train_data = []
    if is_load_all:
        for path in dataset_paths:
            with open(path, 'rb') as file:
                train_data.extend(pickle.load(file))
    else:
        df_use_size = get_use_size_of_each_file(
            dataset_dir=dataset_dir,
            dataset_size=train_size + test_size,
        )

        for path in dataset_paths:
            use_size = df_use_size.loc[path.split('/')[-1], 'use_data_size']
            if use_size != 0:
                with open(path, 'rb') as file:
                    train_data.extend(
                        random.sample(pickle.load(file), use_size)
                    )

    np.random.shuffle(train_data)
    test_data = train_data[:test_size]
    del train_data[:test_size]

    return tuple(train_data), tuple(test_data)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Physical properties calculation conditions'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='train dataset directory',
    )
    parser.add_argument(
        '--train_size',
        type=int,
        default=100000000,  # 100M
        help='test dataset size',
    )
    parser.add_argument(
        '--test_size',
        type=int,
        default=1000000,  # 1M
        help='test dataset size',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        help='save directory for results',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed',
    )

    return parser.parse_args()


def run() -> None:
    args = parse_arguments()
    save_dir = args.save_dir

    if not isdir(save_dir):
        makedirs(save_dir)

    with open(join(save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    fix_seed(args.seed)

    data_sources = ['train', 'test']
    datasets = get_dataset(
        dataset_dir=args.dataset_dir,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    inference_model = InferenceModel()

    for data_source, smiles_list in zip(data_sources, datasets):
        print(f'Start {data_source}')

        pickle.dump(
            smiles_list,
            open(join(save_dir, f'{data_source}_SMILES.pkl'), 'wb'),
        )

        physical_props = get_physical_properties(smiles_list)
        for k in physical_props.keys():
            np.save(
                join(save_dir, f'{data_source}_{k}.npy'),
                physical_props[k],
            )
        del physical_props

        valid_index = np.load(
            join(save_dir, f'{data_source}_valid_index.npy')
        )
        smiles_emb = inference_model.seq_to_emb(smiles_list)[valid_index]
        np.save(
            join(save_dir, f'{data_source}_cddd.npy'),
            np.array(smiles_emb, dtype=np.float32),
        )


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt as e:
        print('[STOP]', e)
