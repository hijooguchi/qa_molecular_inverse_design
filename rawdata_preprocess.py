from os.path import join, isfile, getsize
import os
import argparse
from glob import glob
import gzip
import pandas as pd
import pickle
import re
from typing import Tuple
from rdkit import Chem, RDLogger
from rdkit.Chem import MolToSmiles, MolFromSmiles, GetMolFrags
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem.rdchem import Mol


RDLogger.DisableLog('rdApp.*')

REMOVER = SaltRemover()
ORGANIC_ATOM_SET = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])


def run() -> None:
    args = parse_arguments()
    data_dir = args.dataset_dir
    external_lib_dir = args.external_lib_dir

    print('Start preprocess')
    # convert .smi.gz to pickle
    convert_pickle(data_dir)
    # extract valid SMILES and drop duplicates
    extract_smiles(data_dir)
    # merge additional SMILES and drop duplicates
    merge_additional_data(data_dir)
    # check dataset volume
    check_dataset_volume(data_dir, external_lib_dir)
    print('Finish preprocess')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Physical properties calculation conditions'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='./dataset',
        help='raw data directory',
    )
    parser.add_argument(
        '--external_lib_dir',
        type=str,
        default='./external_lib',
        help='external libraries directory',
    )

    return parser.parse_args()


def convert_pickle(data_dir: str) -> None:
    for i in range(4, 41):
        target_file_list = glob(
            join(data_dir, f'H{str(i).zfill(2)}', '*.smi.gz')
        )

        for file in target_file_list:
            try:
                if isfile(file):
                    if getsize(file) == 0:
                        os.remove(file)
                        print(
                            f'{file} is removed due to its file size is zero.'
                        )
                        continue

                with gzip.open(file, mode='rb') as f:
                    df = pd.read_csv(
                        f,
                        delimiter='\t',
                        usecols=[0],
                        header=None,
                        names=['SMILES'],
                    )

                with open(f'{file[:-7]}.pickle', mode='wb') as f:
                    pickle.dump(tuple(set(df['SMILES'])), f)

                os.remove(file)
                print(f'convert {file} is done.')

            except:
                print(f'convert {file} is skipped.')
                continue


def extract_smiles(data_dir: str) -> None:
    for i in range(4, 41):
        target_file_list = glob(
            join(data_dir, f'H{str(i).zfill(2)}', '*.pickle')
        )

        for file in target_file_list:
            try:
                valid_smiles_list = []
                invalid_smiles_list = []
                invalid_smiles_tranche_list = []
                with open(file, mode='rb') as f:
                    smiles_list = pickle.load(f)

                for smiles in smiles_list:
                    smiles, tranche = preprocess_smiles(smiles, file)

                    if smiles is not None and smiles != '':
                        if tranche == '':
                            valid_smiles_list.append(smiles)
                        else:
                            invalid_smiles_list.append(smiles)
                            invalid_smiles_tranche_list.append(tranche)

                with open(file, mode='wb') as f:
                    pickle.dump(tuple(set(valid_smiles_list)), f)

                tranche_grp_list = set(invalid_smiles_tranche_list.copy())
                for tranche_grp in tranche_grp_list:
                    smiles_saved = []
                    for smiles, tranche in zip(
                        invalid_smiles_list,
                        invalid_smiles_tranche_list,
                    ):
                        if tranche == tranche_grp:
                            smiles_saved.append(smiles)

                    save_name = join(
                        file[:-18],
                        tranche_grp[:3],
                        f'{tranche_grp}_additional.pickle',
                    )
                    if isfile(save_name):
                        with open(save_name, mode='rb') as f:
                            smiles_saved.extend(pickle.load(f))
                    with open(save_name, mode='wb') as f:
                        pickle.dump(tuple(set(smiles_saved)), f)

                print(f'extract {file} is done.')

            except:
                print(f'extract {file} is skipped.')
                continue


def merge_additional_data(data_dir: str) -> None:
    for i in range(4, 41):
        target_file_list = glob(
            join(data_dir, f'H{str(i).zfill(2)}', '*_additional.pickle')
        )

        for file in target_file_list:
            try:
                with open(file, mode='rb') as f:
                    smiles_list = pickle.load(f)

                main_file = re.sub('_additional', '', file)
                print(main_file)
                if isfile(main_file):
                    with open(main_file, mode='rb') as f:
                        smiles_list.extend(pickle.load(f))

                with open(main_file, mode='wb') as f:
                    pickle.dump(tuple(set(smiles_list)), f)

                os.remove(file)
                print(f'merge {file} is done.')

            except:
                print(f'merge {file} is skipped.')
                continue


def check_dataset_volume(data_dir: str, external_lib_dir: str) -> None:
    df_count = []
    for i in range(4, 41):
        target_file_list = sorted(
            glob(
                join(
                    data_dir,
                    f'H{str(i).zfill(2)}',
                    'H[0-9][0-9]?[0-9][0-9][0-9].pickle',
                )
            )
        )

        for file in target_file_list:
            try:
                with open(file, mode='rb') as f:
                    file_count = len(pickle.load(f))

                df_count.append([file[-14:-11], file[-11:-7], file_count])
                print(f'count {file} is done.')

            except:
                print(f'count {file} is skipped.')
                continue

    df_count = pd.DataFrame(
        df_count,
        columns=['n_HeavyAtomCount', 'LogP', 'n_SMILES'],
    )
    df_count.to_csv(
        join(external_lib_dir, 'dataset_detailed_count.csv'),
        encoding='utf-8-sig',
    )


def preprocess_smiles(smiles: str, file_name: str) -> Tuple[str, str]:
    mol = MolFromSmiles(smiles)
    mol = remove_salt_stereo(mol, REMOVER)

    if organic_filter(mol):
        tranche = tranche_validation(mol, file_name)
        fixed_smiles = convert_nonisomeric_canonical_smiles(mol)
    else:
        tranche = ''
        fixed_smiles = ''
    return fixed_smiles, tranche


def remove_salt_stereo(mol: Mol, remover: SaltRemover) -> Mol:
    try:
        mol_strip = remover.StripMol(mol, dontRemoveEverything=True)
        smiles = MolToSmiles(mol_strip, isomericSmiles=False)
        if "." in smiles:
            mol_strip = keep_largest_fragment(mol_strip)

    except:
        mol_strip = None

    return mol_strip


def keep_largest_fragment(mol: Mol) -> Mol:
    mol_frags = GetMolFrags(mol, asMols=True)
    largest_mol = None
    largest_mol_size = 0

    for mol in mol_frags:
        size = mol.GetNumAtoms()
        if size > largest_mol_size:
            largest_mol = mol
            largest_mol_size = size

    return largest_mol


def organic_filter(mol: Mol) -> bool:
    try:
        atom_num_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        is_organic = (set(atom_num_list) <= ORGANIC_ATOM_SET)
        if is_organic:
            return True
        else:
            return False

    except:
        return False


def tranche_validation(mol: Mol, file_name: str) -> str:
    logp = MolLogP(mol)
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    if num_heavy_atoms > 99:
        num_heavy_atoms = 99
    sign = 'M' if logp < 0.0 else 'P'
    tranche = f'H{num_heavy_atoms:02}{sign}{abs(scale_logp_value(logp)):03}'

    if tranche in file_name:
        return ''
    else:
        return tranche


def scale_logp_value(logp: float) -> int:
    if logp < -9.0:
        logp = -9.0
    elif logp > 9.0:
        logp = 9.0

    if logp < 0.0 or logp >= 5.0:
        logp = 100 * int(logp)
    else:
        logp = 10 * int(10 * logp)

    return logp


def convert_nonisomeric_canonical_smiles(mol: Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt as e:
        print('[STOP]', e)
