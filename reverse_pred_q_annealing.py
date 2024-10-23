from dwave.system import LeapHybridSampler
import neal
import numpy as np
from os.path import join, isdir
from os import makedirs
import pickle
from datetime import datetime
import argparse
import json
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from glob import glob
import pandas as pd
import time
from typing import List, Tuple, Dict
from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.Descriptors import MolLogP

from get_props import fix_seed, get_physical_properties
from external_lib.cddd.inference import InferenceModel


RDLogger.DisableLog('rdApp.*')


SOLVER = 'hybrid_binary_quadratic_model_version2p'
# Input on your own token
TOKEN = 'q2Tb-3a852ac5c6fec7dbfa349396f81610177b23c912'
ENDPOINT = 'https://cloud.dwavesys.com/sapi'


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Reverse prediction conditions'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='check point directory of physical property prediction models',
    )
    parser.add_argument(
        '--target_properties',
        type=json.loads,
        default='{"QED": 1.0}',
        help='target properties and values of reverse prediction',
    )
    parser.add_argument(
        '--bit_size_per_constant',
        type=int,
        default=8,
        help='embedding size of each component of cddd',
    )
    parser.add_argument(
        '--emb_size',
        type=int,
        default=512,
        help='embedding size of cddd',
    )
    parser.add_argument(
        '--penalty',
        type=float,
        default=0.0,
        help='penalty for reducing QUBO bit counts',
    )
    parser.add_argument(
        '--iter',
        type=int,
        default=1,
        help='number of iterations to generate new cddd',
    )
    parser.add_argument(
        '--n_sa_sampling',
        type=int,
        default=10000,
        help='number of Simulated Annealing sampling',
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['qa', 'sa'],  # qa: Quantum Annealing, sa: Simulated Annealing
        default='qa',
        help='Annealing method',
    )

    return parser.parse_args()


def pred_cddd_from_props(
    props: Dict[str, float],
    model_dir: str,
    output_dir: str,
    log_filename: str,
    emb_size: int,
    bit_size_per_constant: int = 8,
    penalty: float = 0.0,
    iteration: int = 10,
    n_sa_sampling: int = 1,
    is_qa: bool = True,
) -> None:
    """
    Reverse prediction for desired molecular from physical properties
    by quantum annealing.
    """
    if is_qa:
        sampler = LeapHybridSampler(
            solver=SOLVER,
            token=TOKEN,
            endpoint=ENDPOINT,
        )
    else:
        sampler = neal.SimulatedAnnealingSampler()

    models = {}
    for prop in props.keys():
        models[prop] = pickle.load(
            open(join(model_dir, f'model_{prop}_ss.pkl'), 'rb')
        )
    scalers = {}
    for prop in props.keys():
        scalers[prop] = pickle.load(
            open(join(model_dir, f'standard_scaler_{prop}.pkl'), 'rb')
        )
    std_props = {}
    for prop in props.keys():
        std_props[prop] = scalers[prop].transform(
            np.array(props[prop], dtype=np.float32).reshape(-1, 1)
        )

    total_bits = emb_size * bit_size_per_constant
    qubo = (
        np.zeros(total_bits ** 2, dtype=np.float32)
        .reshape(total_bits, total_bits)
    )

    binary_coef = [
        1 / (2 ** n) for n in range(1, int(bit_size_per_constant) + 1)
    ]
    binary_coef = binary_coef * emb_size

    for prop in std_props.keys():
        coef_origin = models[prop].coef_[0]
        coef = []
        for c in coef_origin:
            coef.extend([c] * bit_size_per_constant)
        beta = models[prop].intercept_ - std_props[prop]

        for i in range(total_bits):
            qubo[i][i] = (
                qubo[i][i] + 2 * beta * coef[i] * 2 * binary_coef[i]
            )
            for c in coef_origin:
                qubo[i][i] = qubo[i][i] - 2 * c * coef[i] * 2 * binary_coef[i]
            for j in range(total_bits):
                qubo[i][j] = (
                    qubo[i][j] +
                    coef[i] * 2 * binary_coef[i] * coef[j] * 2 * binary_coef[j]
                )

    if penalty != 0.0:
        for i in range(total_bits):
            qubo[i][i] = qubo[i][i] + penalty

    pred_cddds = []
    pred_values = {}
    for i in tqdm(range(1, iteration + 1)):
        log_message = f'Iteration {i}/{iteration}'
        print(log_message)
        with open(log_filename, 'a') as f:
            f.write(log_message + '\n')

        if is_qa:
            start = time.time()
            sample = sampler.sample_qubo(qubo)
            end = time.time()
            pred_bit = sample.record[0][0]
        else:
            start = time.time()
            sample = sampler.sample_qubo(qubo, num_reads=n_sa_sampling)
            end = time.time()
            pred_bit = np.array(list(sample.first.sample.values()))

        time_diff = end - start
        log_message = f'  Annealing time:{float(time_diff):.8f},'
        print(log_message)
        with open(log_filename, 'a') as f:
            f.write(log_message + '\n')

        pred_cddd = []
        for start in range(0, total_bits, bit_size_per_constant):
            pred_value = np.array(
                pred_bit[start: start + bit_size_per_constant]
            )
            bin_value = np.array(
                binary_coef[start: start + bit_size_per_constant]
            )
            pred_cddd.append(2 * np.dot(pred_value, bin_value) - 1)
        pred_cddd = np.array(pred_cddd, dtype=np.float32)
        pred_cddds.append(pred_cddd)

        for prop in props.keys():
            if prop not in pred_values.keys():
                pred_values[prop] = np.array([], dtype=np.float32)

            pred = scalers[prop].inverse_transform(
                models[prop].predict(pred_cddd.reshape(1, -1))
            ).reshape(-1)
            pred_values[prop] = np.hstack([pred_values[prop], pred])
            error = pred - props[prop]

            log_message = (
                f'  {prop}: Predicted value:{float(pred):.8f},'
                f' Error:{float(error):.8f}'
            )
            print(log_message)
            with open(log_filename, 'a') as f:
                f.write(log_message + '\n')

        np.save(
            join(output_dir, f'pred_cddd.npy'),
            np.array(pred_cddds, dtype=np.float32),
        )

        for prop in pred_values.keys():
            pred_values[prop] = np.array(pred_values[prop], dtype=np.float32)
        pickle.dump(
            pred_values,
            open(join(output_dir, f'pred_values.pkl'), 'wb'),
        )


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


def get_tranche(smiles_list: List[str]) -> Tuple[List[str], List[str]]:
    valid_smiles_list = []
    tranche_list = []
    for smiles in smiles_list:
        if smiles == '':
            mol = None
        else:
            mol = MolFromSmiles(smiles)

        if mol:
            logp = MolLogP(mol)
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms > 99:
                num_heavy_atoms = 99
            sign = 'M' if logp < 0.0 else 'P'
            valid_smiles_list.append(smiles)
            tranche_list.append(
                f'H{num_heavy_atoms:02}{sign}{abs(scale_logp_value(logp)):03}'
            )

    return valid_smiles_list, tranche_list


def originality_in_all_dataset(
    dataset_dir: str,
    pred_smiles: List[str],
) -> Tuple[List[str], float]:
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

    valid_smiles, tranche_list = get_tranche(pred_smiles)

    df_smiles = pd.DataFrame()
    df_smiles.loc[:, 'SMILES'] = valid_smiles
    df_smiles.loc[:, 'Tranche'] = tranche_list
    df_smiles.loc[:, 'is_origin'] = 0

    target_tranche_list = list(df_smiles['Tranche'].unique())

    for tranche in target_tranche_list:
        pred_smiles_set = set(
            df_smiles.loc[df_smiles['Tranche'] == tranche, 'SMILES'].tolist()
        )
        target_path = join(dataset_dir, f'{tranche[:3]}', f'{tranche}.pickle')
        if target_path in dataset_paths:
            with open(target_path, 'rb') as file:
                all_smiles_set = set(pickle.load(file))

            original_smiles_list = list(pred_smiles_set - all_smiles_set)
        else:
            original_smiles_list = list(pred_smiles_set)

        df_smiles.loc[
            df_smiles['SMILES'].isin(original_smiles_list),
            'is_origin'
        ] = 1

    original_ratio = df_smiles['is_origin'].sum() / len(valid_smiles)

    return valid_smiles, original_ratio


def calc_reverse_pred_score(
    log_filename: str,
    true_values: Dict[str, np.ndarray],
    pred_values: Dict[str, np.ndarray],
    pred_smiles: List[str],
    train_smiles: Tuple[str],
    dataset_dir: str,
    memo: str,
) -> None:
    """
    Calculate following reverse prediction score:
        - Valid SMILES ratio
        - Original SMILES ratio (not in train dataset)
        - Original SMILES ratio (not in all dataset)
        - MAE between target physical properties and physical properties of
          reverse predicted molecular.
    """
    valid_index = true_values.pop('valid_index')
    valid_ratio = len(valid_index) / len(pred_smiles)

    valid_smiles, original_ratio_all = originality_in_all_dataset(
        dataset_dir=dataset_dir,
        pred_smiles=pred_smiles,
    )

    valid_smiles = set(valid_smiles)
    train_smiles = set(train_smiles)
    original_ratio_train = len(valid_smiles - train_smiles) / len(valid_smiles)

    log_message = (
        f'Reverse prediction score ({memo})\n'
        f'  Valid SMILES ratio: {valid_ratio:.5f}\n'
        f'  Origin SMILES ratio (train dataset): {original_ratio_train:.5f}\n'
        f'  Origin SMILES ratio (all dataset): {original_ratio_all:.5f}\n'
    )
    print(log_message)
    with open(log_filename, 'a') as f:
        f.write(log_message)

    for prop in true_values.keys():
        log_message = (
            f'  {prop}: Mean {np.mean(true_values[prop]):.5f}, '
            f'SD {np.std(true_values[prop]):.5f}'
        )

        if prop in pred_values.keys():
            mae = mean_absolute_error(
                true_values[prop],
                pred_values[prop][valid_index],
            )
            log_message += f', MAE {mae:.5f}\n'
        else:
            log_message += f'\n'

        print(log_message)
        with open(log_filename, 'a') as f:
            f.write(log_message)


def extract_max_len_valid_smiles(smiles_list: List[str]) -> List[str]:
    extracted_smiles_list = []
    for smiles in smiles_list:
        decoded_smiles = ''
        smiles_len = 0
        for start in range(0, len(smiles)):
            for end in range(1, len(smiles) + 1):
                text = smiles[start:end]
                mol = MolFromSmiles(text)

                if mol is not None:
                    text_len = len(text)
                    if text_len > smiles_len:
                        decoded_smiles = text
                        smiles_len = text_len

        extracted_smiles_list.append(decoded_smiles)

    return extracted_smiles_list


def convert_canonical_smiles(smiles_list: List[str]) -> List[str]:
    canonical_smiles_list = []
    for smiles in smiles_list:
        if smiles == '':
            mol = None
        else:
            mol = MolFromSmiles(smiles)

        if mol:
            canonical_smiles = MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=False,
            )
        else:
            canonical_smiles = ''

        canonical_smiles_list.append(canonical_smiles)

    return canonical_smiles_list


def run() -> None:
    """
    Reverse prediction for desired molecular from physical properties.
    """
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    args = parse_arguments()
    checkpoint_dir = args.checkpoint_dir
    is_qa = True if args.method == 'qa' else False

    if is_qa:
        output_dir = join(checkpoint_dir, 'reverse_pred', current_time)
    else:
        output_dir = join(checkpoint_dir, 'reverse_pred/sa', current_time)

    log_filename = join(output_dir, 'log.txt')

    if not isdir(checkpoint_dir):
        raise FileNotFoundError(f'{checkpoint_dir} is not available')
    else:
        makedirs(output_dir)

    with open(join(output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    with open(join(checkpoint_dir, 'config.json'), 'r') as f:
        args_train = argparse.Namespace(**json.load(f))

    with open(log_filename, 'w') as f:
        f.write(f'Reverse prediction log: created at {current_time} \n')

    fix_seed(args_train.seed)

    print("Start solving.")
    pred_cddd_from_props(
        props=args.target_properties,
        model_dir=checkpoint_dir,
        output_dir=output_dir,
        log_filename=log_filename,
        emb_size=args.emb_size,
        bit_size_per_constant=args.bit_size_per_constant,
        penalty=args.penalty,
        iteration=args.iter,
        is_qa=is_qa,
        n_sa_sampling=args.n_sa_sampling,
    )

    print("Converting CDDD to SMILES.")
    inference_model = InferenceModel()
    pred_smiles = inference_model.emb_to_seq(
        np.load(join(output_dir, 'pred_cddd.npy'))
    )
    if type(pred_smiles) not in [list, tuple]:
        pred_smiles = [pred_smiles]

    pickle.dump(
        pred_smiles,
        open(join(output_dir, f'pred_SMILES.pkl'), 'wb'),
    )
    fixed_pred_smiles = extract_max_len_valid_smiles(pred_smiles)
    pickle.dump(
        fixed_pred_smiles,
        open(join(output_dir, f'fixed_pred_SMILES.pkl'), 'wb'),
    )

    pred_canonical_smiles = convert_canonical_smiles(pred_smiles)
    pickle.dump(
        pred_canonical_smiles,
        open(join(output_dir, f'pred_canonical_SMILES.pkl'), 'wb'),
    )

    fixed_pred_canonical_smiles = convert_canonical_smiles(fixed_pred_smiles)
    pickle.dump(
        fixed_pred_canonical_smiles,
        open(join(output_dir, f'fixed_pred_canonical_SMILES.pkl'), 'wb'),
    )

    print("Calculating scores.")
    true_values = get_physical_properties(pred_smiles)
    pickle.dump(
        true_values,
        open(join(output_dir, f'true_values.pkl'), 'wb'),
    )
    fixed_true_values = get_physical_properties(fixed_pred_smiles)
    pickle.dump(
        fixed_true_values,
        open(join(output_dir, f'fixed_true_values.pkl'), 'wb'),
    )

    train_smiles = pickle.load(
        open(join(checkpoint_dir, f'train_SMILES.pkl'), 'rb')
    )
    pred_values = pickle.load(
        open(join(output_dir, f'pred_values.pkl'), 'rb')
    )

    calc_reverse_pred_score(
        log_filename=log_filename,
        true_values=pickle.load(
            open(join(output_dir, f'true_values.pkl'), 'rb')
        ),
        pred_values=pred_values,
        pred_smiles=pred_smiles,
        train_smiles=train_smiles,
        dataset_dir=args_train.dataset_dir,
        memo='non fixed',
    )
    calc_reverse_pred_score(
        log_filename=log_filename,
        true_values=pickle.load(
            open(join(output_dir, f'true_values.pkl'), 'rb')
        ),
        pred_values=pred_values,
        pred_smiles=pred_canonical_smiles,
        train_smiles=train_smiles,
        dataset_dir=args_train.dataset_dir,
        memo='non fixed canonical',
    )
    calc_reverse_pred_score(
        log_filename=log_filename,
        true_values=pickle.load(
            open(join(output_dir, f'fixed_true_values.pkl'), 'rb')
        ),
        pred_values=pred_values,
        pred_smiles=fixed_pred_smiles,
        train_smiles=train_smiles,
        dataset_dir=args_train.dataset_dir,
        memo='fixed',
    )
    calc_reverse_pred_score(
        log_filename=log_filename,
        true_values=pickle.load(
            open(join(output_dir, f'fixed_true_values.pkl'), 'rb')
        ),
        pred_values=pred_values,
        pred_smiles=fixed_pred_canonical_smiles,
        train_smiles=train_smiles,
        dataset_dir=args_train.dataset_dir,
        memo='fixed canonical',
    )
    print('Finish solving.')


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt as e:
        print('[STOP]', e)
